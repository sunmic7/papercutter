"""Isolated data and HTTP service for the browser regression suite."""
import base64
import io
import json
import sys
from pathlib import Path
from http.server import ThreadingHTTPServer
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_review_beta.server import ReviewStore, ReviewHandler, PROJECT_FILENAME

root = Path(sys.argv[1])
store = ReviewStore(root / "data", projects_root=root / "projects", staff_data_root=root / "staff")
password = "review-ui-test-only"
super_admin = store.create_user("qa-super", password, "super_admin")
admin = store.create_managed_user(super_admin, "qa-admin", password, "admin")
global_admin = store.create_managed_user(super_admin, "qa-global-admin", password, "admin")
reviewer = store.create_managed_user(admin, "qa-reviewer", password, "reviewer")
marker = store.create_managed_user(admin, "qa-marker", password, "annotator")
other_marker = store.create_managed_user(admin, "qa-marker-other", password, "annotator")
store.set_responsible_person(super_admin,other_marker['id'],None)
inactive = store.create_managed_user(admin, "qa-inactive", password, "annotator")
store.manage_user(admin, inactive["id"], {"active": False})
store.set_responsible_person(admin, marker["id"], reviewer["id"])
image = Image.new("RGB", (1000, 1400), "#fffefa")
draw = ImageDraw.Draw(image)
draw.text((70, 60), "PAPERFLOW REVIEW - MATHEMATICS", fill="#243859", font_size=34)
for index in range(1, 9):
    top = 120 + index * 130
    draw.text((80, top), f"{index}.   12 + {index} = (     )", fill="#1e293b", font_size=28)
    draw.line((80, top + 65, 900, top + 65), fill="#d1d8e5", width=1)
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
preview = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
source = root / 'originals' / '数学待派发教材'
source.mkdir(parents=True, exist_ok=True)
image.save(source / '1.jpg')
store.library.source_root = source.parent
store.library.scan(super_admin)
new_book = store.library.listing(super_admin, catalog=True)[0]['id']
def document(book_id, name):
    return {"schema_version": 1, "book": {"id": book_id, "name": name},
        "project_revision": 1, "snapshot_fingerprint": "a" * 64,
        "project_document": {"project_revision": 1, "pages": []},
        "pages": [{"page_key": f"question-{index}", "kind": "question", "page_index": index,
            "image_name": f"page-{index+1}.jpg", "width": 1000, "height": 1400,
            "regions": [{"id": f"q-{index}", "display_number": f"1-{index+1}", "bbox": [65,230,920,340]}],
            "preview": preview} for index in range(3)]}
project = {"book_root": str(root / "original"), "project_revision": 1, "language":"zh",
    "stage":"小学", "grade":"六年级", "subject":"数学", "pages":[{"image":"page-01.jpg",
    "status":"accepted", "questions":[{"id":"q-1", "number_path":["1","1"],"bbox":[65,230,920,340],"text":"12+1=()"}]}],
    "answer_pages":[{"image":"answer-01.jpg","status":"accepted","answers":[{"id":"a-1","text":"13","bbox":[1,2,100,200]}]}],
    "pairs":[{"id":"pair-1","question_id":"q-1","answer_ids":["a-1"],"status":"accepted"}]}
target=root/"projects"/"BQAWORKSPACE"
target.mkdir(parents=True)
(target/PROJECT_FILENAME).write_text(json.dumps(project,ensure_ascii=False),encoding="utf-8")
store.assign_project(admin,"BQAWORKSPACE",marker["id"])
old=store.create_submission(marker,document("BQAWORKSPACE","六年级数学 · 流程测试"))
latest=store.create_submission(marker,document("BQAWORKSPACE","六年级数学 · 流程测试"))
other=store.create_submission(other_marker,document("BQAOTHER","其他负责人的教材"))
foreign_target=root/'projects'/'BQAOTHER';foreign_target.mkdir()
(foreign_target/PROJECT_FILENAME).write_text(json.dumps(project,ensure_ascii=False),encoding='utf-8')
store.assign_project(admin,'BQAOTHER',other_marker['id'])
ReviewHandler.store=store
ReviewHandler.static_root=ROOT/"online_review_beta/static"
server=ThreadingHTTPServer(("127.0.0.1",0),ReviewHandler)
config={"server":f"http://127.0.0.1:{server.server_port}", "password":password,
    "new_book_id":new_book, "marker_id":marker['id'], "originals":str(source.parent.resolve()),
    "accounts":{"super_admin":"qa-super","admin":"qa-global-admin","reviewer":"qa-reviewer","annotator":"qa-marker"},
    "old_id":old["id"], "task_id":latest["id"], "other_id":other["id"], "book_id":"BQAWORKSPACE"}
(root/"config.json").write_text(json.dumps(config),encoding="utf-8")
print("Isolated review workspace ready.",flush=True)
server.serve_forever()
