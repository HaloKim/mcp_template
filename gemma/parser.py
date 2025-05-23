

import json
import re
from collections.abc import Sequence
from typing import Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("gemma")
class GemmaToolParser(ToolParser):
    """
    Tool call parser for Gemma models intended for use with the
    examples/tool_chat_template_gemma.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser gemma are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        요청을 조정하여 특수 토큰을 유지하게 합니다.
        """
        if request.tools and request.tool_choice != 'none':
            request.skip_special_tokens = False
        return request# SPDX-License-Identifier: Apache-2.0
    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        # 코드 블록으로 감싸진 도구 호출 처리 (```json [...] ``` 형식)
        if "```json" in model_output:
            # 일반 텍스트로 처리
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)
                                                
        # 정규식으로 JSON 배열 패턴 찾기
        json_pattern = r'\[\s*{\s*"name"\s*:.*}\s*\]'
        json_matches = re.search(json_pattern, model_output, re.DOTALL)
        
        if not json_matches:
            json_pattern = r'\[\s*{\s*\'name\'\s*:.*}\s*\]'
            json_matches = re.search(json_pattern, model_output, re.DOTALL)
                    
        if not json_matches:
            return ExtractedToolCallInformation(tools_called=False,
                                               tool_calls=[],
                                               content=model_output)
                
        try:
            # JSON 문자열 파싱
            json_str = json_matches.group(0).replace("'", '"')
            
            # 기본 검사 - 중괄호 쌍 확인
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            if open_braces == 0 or open_braces != close_braces:
                # 완전한 JSON이 아님
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)
            
            function_call_arr = json.loads(json_str)
            if not isinstance(function_call_arr, list):
                function_call_arr = [function_call_arr]
                
            tool_calls: list[ToolCall] = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(raw_function_call["arguments"])))
                for raw_function_call in function_call_arr
            ]

            # JSON 도구 호출 전의 내용 추출
            content = model_output[:json_matches.start()].strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None)

        except Exception as e:
            logger.exception(f"Error in extracting tool call from response: {str(e)}")
            # return information to just treat the tool call as regular JSON
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        # 일반 텍스트인 경우 그대로 반환
        if '[' not in current_text or '{' not in current_text:
            return DeltaMessage(content=delta_text)

        try:
            # JSON 형식으로 보이는 부분 검색
            start_idx = current_text.find('[')
            if start_idx < 0:
                return DeltaMessage(content=delta_text)
                
            json_str = current_text[start_idx:]
            
            # 코드 블록으로 감싸진 경우 처리 (```json [...] ``` 형식)
            if json_str.startswith("[```"):
                # 아직 제대로 된 JSON이 아님, 일반 텍스트로 간주
                return DeltaMessage(content=delta_text)
            
            # 기본 JSON 구문 검사 - 중괄호 쌍 확인
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            if open_braces == 0 or open_braces != close_braces:
                # 완전한 JSON이 아님, 일반 텍스트로 간주
                return DeltaMessage(content=delta_text)
            
            # JSON 일부분만 추출
            try:
                # 정규식을 사용하여 JSON 배열 패턴 찾기
                json_pattern = r'\[\s*{\s*"name"\s*:.*}\s*\]'
                json_matches = re.search(json_pattern, json_str, re.DOTALL)
                
                if not json_matches:
                    json_pattern = r'\[\s*{\s*\'name\'\s*:.*}\s*\]'
                    json_matches = re.search(json_pattern, json_str, re.DOTALL)
                    
                if not json_matches:
                    # JSON 패턴 찾지 못함
                    return DeltaMessage(content=delta_text)
                    
                json_extract = json_matches.group(0)
                json_extract = json_extract.replace("'", '"')
                
                # JSON 파싱
                tool_call_arr = json.loads(json_extract)
                if not isinstance(tool_call_arr, list):
                    tool_call_arr = [tool_call_arr]
                
                # 도구 호출이 없으면 일반 텍스트로 처리
                if len(tool_call_arr) == 0:
                    return DeltaMessage(content=delta_text)
                
                # 도구 호출 처리 시작
                if self.current_tool_id < 0:
                    # 첫 번째 도구 호출 시작
                    self.current_tool_id = 0
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool = [""] * len(tool_call_arr)
                    self.prev_tool_call_arr = [{} for _ in range(len(tool_call_arr))]
                
                # 현재 처리할 도구 가져오기
                if self.current_tool_id >= len(tool_call_arr):
                    return None
                    
                current_tool_call = tool_call_arr[self.current_tool_id]
                
                # 도구 이름 처리
                if not self.current_tool_name_sent:
                    function_name = current_tool_call.get("name")
                    if function_name:
                        # 도구 이름 전송
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=f"chatcmpl-tool-{random_uuid()}",
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True)
                            )
                        ])
                        self.current_tool_name_sent = True
                        
                        # 현재 도구 정보 저장
                        if self.current_tool_id < len(self.prev_tool_call_arr):
                            self.prev_tool_call_arr[self.current_tool_id] = {"name": function_name}
                        else:
                            self.prev_tool_call_arr.append({"name": function_name})
                            
                        return delta
                    return None
                
                # 도구 인자 처리
                cur_arguments = current_tool_call.get("arguments")
                if not cur_arguments:
                    return None
                
                # 이전에 보낸 인자 정보 가져오기
                if self.current_tool_id < len(self.prev_tool_call_arr):
                    prev_tool = self.prev_tool_call_arr[self.current_tool_id]
                    prev_arguments = prev_tool.get("arguments")
                else:
                    prev_arguments = None
                
                # 인자 처리
                if not prev_arguments:
                    # 첫 인자 전송
                    cur_args_json = json.dumps(cur_arguments)
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=cur_args_json
                            ).model_dump(exclude_none=True)
                        )
                    ])
                    
                    # 상태 업데이트
                    if self.current_tool_id < len(self.streamed_args_for_tool):
                        self.streamed_args_for_tool[self.current_tool_id] = cur_args_json
                    else:
                        self.streamed_args_for_tool.append(cur_args_json)
                        
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = cur_arguments
                    else:
                        self.prev_tool_call_arr.append({"name": current_tool_call.get("name"), "arguments": cur_arguments})
                    
                    return delta
                
                return None
                
            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"JSON 파싱 오류: {str(e)}")
                return DeltaMessage(content=delta_text)
                
        except Exception as e:
            logger.error(f"도구 호출 처리 중 오류: {str(e)}")
            return DeltaMessage(content=delta_text)