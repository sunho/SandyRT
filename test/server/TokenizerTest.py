import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sandy_server.tokenizer import encode_messages


class FakeTokenizer:
    chat_template = "template"

    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return [11, 12]


class TokenizerTest(unittest.TestCase):
    def test_passes_request_chat_template_kwargs(self):
        tokenizer = FakeTokenizer()

        ids = encode_messages(
            tokenizer,
            [{"role": "user", "content": "hello"}],
            {"enable_thinking": False},
        )

        self.assertEqual(ids, [11, 12])
        self.assertEqual(tokenizer.kwargs["enable_thinking"], False)
        self.assertEqual(tokenizer.kwargs["tokenize"], True)
        self.assertEqual(tokenizer.kwargs["add_generation_prompt"], True)

    def test_rejects_reserved_chat_template_kwargs(self):
        with self.assertRaisesRegex(ValueError, "cannot override: tokenize"):
            encode_messages(
                FakeTokenizer(),
                [{"role": "user", "content": "hello"}],
                {"tokenize": False},
            )


if __name__ == "__main__":
    unittest.main()
