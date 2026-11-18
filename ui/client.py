"""
PathSnap: Client UI Runner (Laptop 2)
Serves the Evaluator UI dashboard on Laptop 2 (port 3000).
Allows connecting to Laptop 1 backend IP address.
"""

import http.server
import socketserver
import os
import webbrowser

PORT = 3000
UI_DIR = os.path.join(os.path.dirname(__file__), "templates")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

def start_ui():
    print("=" * 60)
    print(f"STARTING PATHSNAP EVALUATOR UI (LAPTOP 2)")
    print(f"Serving dashboard at: http://localhost:{PORT}")
    print("=" * 60)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down UI client...")

if __name__ == "__main__":
    start_ui()
