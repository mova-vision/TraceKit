"""Tiny static server for the tools folder (zone picker, future docs/pages).

    python tools/server.py            # http://127.0.0.1:8000/zone_picker.html
    python tools/server.py 8080       # custom port
"""

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))

handler = http.server.SimpleHTTPRequestHandler
print(f"serving tools/ at http://127.0.0.1:{PORT}/zone_picker.html")
http.server.HTTPServer(("127.0.0.1", PORT), handler).serve_forever()
