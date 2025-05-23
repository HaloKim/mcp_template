# SPDX-License-Identifier: Apache-2.0

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
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("custom_function")
class CustomFunctionParser(ToolParser):
    """
    Tool call parser for the custom format: <function=function_name>{parameters}</function>
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        
        # Regex pattern to match <function=function_name>{...}</function>
        self.tool_call_regex = re.compile(
            r'<function=([^>]+)>(.*?)</function>', 
            re.DOTALL
        )
        
        # For streaming
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self.partial_tool_calls = ""

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Check if there's a function call in the output
        if '<function=' not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

        try:
            # Find all function calls
            matches = self.tool_call_regex.findall(model_output)
            
            if not matches:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            tool_calls = []
            for function_name, arguments_str in matches:
                try:
                    # Parse the JSON arguments
                    if arguments_str.strip():
                        arguments = json.loads(arguments_str)
                    else:
                        arguments = {}
                    
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=function_name,
                                # arguments는 항상 JSON 문자열이어야 함
                                arguments=json.dumps(arguments, ensure_ascii=False)
                            )
                        )
                    )
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse arguments for function {function_name}: {arguments_str}")
                    # 파싱 실패 시 빈 객체로 처리
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=function_name,
                                arguments="{}"
                            )
                        )
                    )
            
            # Extract content before the first function call
            first_function_index = model_output.find('<function=')
            content = model_output[:first_function_index].strip() if first_function_index > 0 else None
            
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None
            )
            
        except Exception as e:
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

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
        
        # If no function call pattern detected yet, stream as content
        if '<function=' not in current_text:
            return DeltaMessage(content=delta_text)
        
        # Check if we're in the middle of a function call
        if '<function=' in current_text and '</function>' not in current_text:
            # We're in the middle of streaming a function call
            # Don't send anything until we have the complete call
            return None
        
        try:
            # Find complete function calls
            matches = self.tool_call_regex.findall(current_text)
            
            if not matches:
                return None
            
            # Check if we have new complete function calls
            current_num_calls = len(matches)
            prev_num_calls = len(self.prev_tool_call_arr)
            
            if current_num_calls > prev_num_calls:
                # We have a new complete function call
                latest_match = matches[-1]
                function_name, arguments_str = latest_match
                
                try:
                    if arguments_str.strip():
                        arguments = json.loads(arguments_str)
                    else:
                        arguments = {}
                    
                    # Send the new tool call
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=current_num_calls - 1,
                                type="function",
                                id=f"chatcmpl-tool-{random_uuid()}",
                                function=DeltaFunctionCall(
                                    name=function_name,
                                    # arguments는 항상 JSON 문자열이어야 함
                                    arguments=json.dumps(arguments, ensure_ascii=False)
                                ).model_dump(exclude_none=True)
                            )
                        ]
                    )
                    
                    # Update state
                    self.prev_tool_call_arr.append({
                        "name": function_name,
                        "arguments": arguments
                    })
                    
                    return delta
                    
                except json.JSONDecodeError:
                    # Arguments not yet complete, send with empty args
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=current_num_calls - 1,
                                type="function",
                                id=f"chatcmpl-tool-{random_uuid()}",
                                function=DeltaFunctionCall(
                                    name=function_name,
                                    arguments="{}"
                                ).model_dump(exclude_none=True)
                            )
                        ]
                    )
                    
                    self.prev_tool_call_arr.append({
                        "name": function_name,
                        "arguments": {}
                    })
                    
                    return delta
            
            # Check if we're streaming content after function calls
            last_function_end = current_text.rfind('</function>')
            if last_function_end != -1:
                content_after = current_text[last_function_end + len('</function>'):]
                if content_after and content_after in delta_text:
                    return DeltaMessage(content=content_after)
            
            return None
            
        except Exception as e:
            logger.exception("Error in streaming tool call extraction")
            return None