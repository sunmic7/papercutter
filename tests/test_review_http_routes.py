"""审核网站 HTTP 处理层的静态护栏。

``ReviewHandler`` 是 ``BaseHTTPRequestHandler`` 的子类，只有 ``store`` 与
``static_root`` 两个由模块底部注入的属性。业务对象一律挂在 ``self.store`` 上，
例如 ``self.store.library``。曾经有一处 ``self.library.diagnose_delivery(...)``
逃过了全部单元测试（那些测试直接调 store），直到真机点「检查缺什么」才暴露成
400。这里用 AST 把这类错误挡在提交之前。
"""
import ast
import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "online_review_beta" / "server.py"

# BaseHTTPRequestHandler 提供的成员，以及模块底部注入的两个类属性。
INHERITED = {
    "rfile", "wfile", "headers", "path", "requestline", "request_version",
    "client_address", "server", "command", "close_connection", "raw_requestline",
    "error_message_format", "error_content_type", "protocol_version", "server_version",
    "sys_version", "default_request_version", "responses", "timeout",
    "send_response", "send_response_only", "send_header", "end_headers",
    "send_error", "flush_headers", "handle", "handle_one_request", "log_request",
    "log_error", "log_message", "address_string", "version_string", "date_time_string",
    "handle_expect_100", "parse_request", "finish",
}
INJECTED = {"store", "static_root"}


class ReviewHandlerAttributeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SERVER.read_text(encoding="utf-8"))
        handlers = [node for node in cls.tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "ReviewHandler"]
        assert handlers, "server.py 里找不到 ReviewHandler"
        cls.handler = handlers[0]

    def test_handler_never_touches_undefined_self_attributes(self):
        assigned = set()
        used = {}
        for node in ast.walk(self.handler):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "self"):
                if isinstance(node.ctx, ast.Store):
                    assigned.add(node.attr)
                else:
                    used.setdefault(node.attr, node.lineno)
        assigned |= {item.name for item in self.handler.body
                     if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
        unknown = {name: line for name, line in used.items()
                   if name not in assigned and name not in INHERITED and name not in INJECTED}
        self.assertEqual(
            {}, unknown,
            "ReviewHandler 使用了不存在的属性（应该在 self.store 上）："
            + ", ".join(f"line {line} self.{name}" for name, line in sorted(unknown.items())))

    def test_diagnosis_route_uses_store_library(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("self.store.library.diagnose_delivery(", source)
        self.assertNotIn("self.library.diagnose_delivery(", source)

    def test_delivery_preview_routes_exist(self):
        source = SERVER.read_text(encoding="utf-8")
        for route in ('"diagnosis"', '"dataset"', '"samples"', '"sample-image"', '"notes"'):
            self.assertIn(route, source)


if __name__ == "__main__":
    unittest.main()
