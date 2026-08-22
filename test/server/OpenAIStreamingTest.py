import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fastapi.testclient import TestClient

import sandy_server.app as app_module
from sandy_server.config import ServerConfig
from sandy_server.grpc_client import sandy_inference_pb2


class FakeTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        return [1]

    def decode(self, ids, **_kwargs):
        values = {
            (10,): "Hello",
            (10, 11): "Hello world",
        }
        return values.get(tuple(ids), "")


class FakeCall:
    def __init__(self, events):
        self._events = iter(events)
        self.cancelled = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._events)

    def cancel(self):
        self.cancelled = True


class FakeGrpcClient:
    def __init__(self, _target):
        self.call = None
        self.stream_kwargs = None

    def generate_stream(self, **kwargs):
        self.stream_kwargs = kwargs
        first = sandy_inference_pb2.GenerateStreamResponse(request_id="id")
        first.token.token_id = 10
        second = sandy_inference_pb2.GenerateStreamResponse(request_id="id")
        second.token.token_id = 11
        done = sandy_inference_pb2.GenerateStreamResponse(request_id="id")
        done.done.finish_reason = "stop"
        done.done.prompt_tokens = 3
        done.done.completion_tokens = 2
        self.call = FakeCall([first, second, done])
        return self.call


class OpenAIStreamingTest(unittest.TestCase):
    def test_streams_openai_chunks_usage_and_done(self):
        fake_grpc = FakeGrpcClient("unused")
        with (
            patch.object(app_module, "load_tokenizer", return_value=FakeTokenizer()),
            patch.object(app_module, "SandyGrpcClient", return_value=fake_grpc),
        ):
            app = app_module.create_app(ServerConfig(
                model_id="test-model",
                tokenizer_path=Path("unused"),
                grpc_target="unused",
                eos_token_id=1,
                end_of_turn_token_id=106,
                tool_response_token_id=50,
            ))

        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        self.assertEqual(response.status_code, 200)
        payloads = [
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(payloads[-1], "[DONE]")
        chunks = [json.loads(payload) for payload in payloads[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        content = "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
            if chunk["choices"]
        )
        self.assertEqual(content, "Hello world")
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["total_tokens"], 5)
        self.assertTrue(fake_grpc.call.cancelled)
        self.assertEqual(fake_grpc.stream_kwargs["stop_token_ids"], [1, 106, 50])


if __name__ == "__main__":
    unittest.main()
