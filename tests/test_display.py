import unittest
from unittest.mock import patch

from flask import render_template_string
from demo_grounded_search import GroundedSpan
from web_app import app, annotate_answer_html


class DisplayTests(unittest.TestCase):
    def test_unicode_offsets_and_omitted_start(self):
        text = '中文😀 Gemini 回答'
        target = 'Gemini'
        start = len('中文😀 '.encode('utf-8'))
        span = GroundedSpan(target, start, start + len(target.encode()), [2])
        result = annotate_answer_html(text, [span])
        self.assertIn('中文😀 <mark', result)
        self.assertIn('>Gemini<sup', result)
        first = GroundedSpan('中文', None, len('中文'.encode()), [0])
        self.assertIn('>中文<sup', annotate_answer_html(text, [first]))

    def test_invalid_offsets_fall_back_and_escape_html(self):
        span = GroundedSpan('正文', 2, 4, [1])
        result = annotate_answer_html('<script>正文</script>', [span])
        self.assertIn('&lt;script&gt;', result)
        self.assertIn('>正文<sup', result)

    def test_json_is_unicode_and_html_safe(self):
        with app.app_context():
            rendered = render_template_string('{{ value | tojson(indent=2) }}', value={'text': '中文</script>'})
        self.assertIn('中文', rendered)
        self.assertNotIn('\\u4e2d', rendered)
        self.assertNotIn('</script>', rendered)

    def test_stream_client_failure_is_sse_error(self):
        with patch('web_app.get_client', side_effect=RuntimeError('test failure')):
            response = app.test_client().post('/ask/stream', data={'question': '测试'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('event: error', response.get_data(as_text=True))
        self.assertIn('test failure', response.get_data(as_text=True))

    def test_index_renders(self):
        response = app.test_client().get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('overflow-wrap: anywhere', response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
