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


@ToolParserManager.register_module("midm")
class MidmToolParser(ToolParser):
    """
    Tool call parser for Mi:dm model format: 
    <tool_call>
    {"name": "function_name", "arguments": {...}}
    </tool_call>
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        
        # Regex pattern to match <tool_call>\n{...}\n</tool_call>
        self.tool_call_regex = re.compile(
            r'<tool_call>\s*(.*?)\s*</tool_call>', 
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
        # Check if there's a tool call in the output
        if '<tool_call>' not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

        try:
            # Find all tool calls
            matches = self.tool_call_regex.findall(model_output)
            
            if not matches:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            tool_calls = []
            for tool_call_str in matches:
                try:
                    # Parse the JSON tool call
                    tool_call_data = json.loads(tool_call_str.strip())
                    
                    function_name = tool_call_data.get("name", "")
                    arguments = tool_call_data.get("arguments", {})
                    
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
                    logger.error(f"Failed to parse tool call: {tool_call_str}")
                    continue
            
            # Extract content before the first tool call
            first_tool_call_index = model_output.find('<tool_call>')
            content = model_output[:first_tool_call_index].strip() if first_tool_call_index > 0 else None
            
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
        
        # If no tool call pattern detected yet, stream as content
        if '<tool_call>' not in current_text:
            return DeltaMessage(content=delta_text)
        
        # Check if we're in the middle of a tool call
        if '<tool_call>' in current_text and '</tool_call>' not in current_text:
            # We're in the middle of streaming a tool call
            # Don't send anything until we have the complete call
            return None
        
        try:
            # Find complete tool calls
            matches = self.tool_call_regex.findall(current_text)
            
            if not matches:
                return None
            
            # Check if we have new complete tool calls
            current_num_calls = len(matches)
            prev_num_calls = len(self.prev_tool_call_arr)
            
            if current_num_calls > prev_num_calls:
                # We have a new complete tool call
                latest_match = matches[-1]
                
                try:
                    tool_call_data = json.loads(latest_match.strip())
                    function_name = tool_call_data.get("name", "")
                    arguments = tool_call_data.get("arguments", {})
                    
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
                                    name="",
                                    arguments="{}"
                                ).model_dump(exclude_none=True)
                            )
                        ]
                    )
                    
                    self.prev_tool_call_arr.append({
                        "name": "",
                        "arguments": {}
                    })
                    
                    return delta
            
            # Check if we're streaming content after tool calls
            last_tool_call_end = current_text.rfind('</tool_call>')
            if last_tool_call_end != -1:
                content_after = current_text[last_tool_call_end + len('</tool_call>'):]
                if content_after and content_after in delta_text:
                    return DeltaMessage(content=content_after)
            
            return None
            
        except Exception as e:
            logger.exception("Error in streaming tool call extraction")
            return None
