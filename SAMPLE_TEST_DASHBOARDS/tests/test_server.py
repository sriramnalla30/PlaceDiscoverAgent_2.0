import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server  # noqa: E402


class PhoneInfo(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(server.phone_info('09019522282'), {'kind': 'mobile', 'e164': '+919019522282'})
        self.assertEqual(server.phone_info('04069146071'), {'kind': 'landline', 'e164': '+914069146071'})
        self.assertEqual(server.phone_info('08041144999')['kind'], 'landline')
        self.assertEqual(server.phone_info('18001031010'), {'kind': 'toll-free', 'e164': '+9118001031010'})
        self.assertEqual(server.phone_info(''), {'kind': 'none', 'e164': ''})

    def test_shared_numbers(self):
        places = server.enrich([{'phone': '09019522282'}, {'phone': '09019522282'}, {'phone': '09845633625'}, {}])
        self.assertEqual([p['_phone_shared'] for p in places], [2, 2, 1, 0])


class Validation(unittest.TestCase):
    def ok(self, **kw):
        return server.validate({'params': {'query': 'gyms in Koramangala Bangalore', 'lat': '12.9', 'lng': '77.6', **kw}})

    def test_valid(self):
        self.assertEqual(self.ok(radius='2000', lang='hi')['radius'], 2000)

    def test_rejects(self):
        for bad in ({'query': ''}, {'query': 'a\nb'}, {'lat': 'x'}, {'lat': '95'}, {'radius': '5'}, {'radius': '1e3'}, {'lang': 'x; rm'}):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.ok(**bad)


class Execute(unittest.TestCase):
    def run_with(self, output, returncode=0):
        def fake_run(command, cwd, capture_output, timeout):
            Path(command[command.index('-results') + 1]).write_text(output, encoding='utf-8')
            return mock.Mock(returncode=returncode, stdout=b'', stderr=b'')
        with mock.patch.object(server.subprocess, 'run', fake_run), mock.patch.object(server.BINARY.__class__, 'exists', lambda self: True):
            return server.execute({'params': {'query': 'gyms in X', 'lat': '12.9', 'lng': '77.6'}})

    def test_parses_places_and_metrics(self):
        rows = json.dumps([{'title': 'A', 'phone': '09019522282', 'web_site': 'https://a.example'}, {'title': 'B', 'phone': ''}])
        result = self.run_with(rows + '\n')
        self.assertTrue(result['ok'])
        self.assertEqual(result['metrics']['places'], 2)
        self.assertEqual(result['metrics']['with_phone'], 1)
        self.assertEqual(result['metrics']['mobile'], 1)

    def test_empty_output_explained(self):
        result = self.run_with('')
        self.assertFalse(result['ok'])
        self.assertEqual(result['error_kind'], 'empty')

    def test_null_entries_are_skipped(self):
        for output in ('null\n', '[null]\n', json.dumps([None, {'title': 'A', 'phone': ''}]) + '\n'):
            places = server.parse_output(output)
            self.assertTrue(all(isinstance(p, dict) for p in places))
        result = self.run_with('null\n')
        self.assertFalse(result['ok'])
        self.assertEqual(result['error_kind'], 'empty')

    def test_command_has_no_shell_and_pin(self):
        cmd = server.build_command({'query': 'q', 'lat': 1.5, 'lng': 2.5, 'radius': 900, 'lang': 'hi'}, 'q.txt', 'o.json')
        self.assertIn('-fast-mode', cmd)
        self.assertEqual(cmd[cmd.index('-geo') + 1], '1.5,2.5')
        self.assertEqual(cmd[cmd.index('-radius') + 1], '900')


class History(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(server, 'HISTORY_DB', Path(d) / 'h.sqlite3'):
                saved = server.save_history({'ok': True, 'params': {'query': 'q'}, 'elapsed_ms': 5, 'metrics': {'places': 1}, 'places': [{'title': 'A'}]})
                self.assertEqual(server.get_history()['total'], 1)
                self.assertEqual(server.get_run(saved['id'])['places'][0]['title'], 'A')


if __name__ == '__main__':
    unittest.main()
