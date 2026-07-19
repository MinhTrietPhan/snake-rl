from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


SOURCE = Path(r"C:\Users\LENOVO\Desktop\Snake_AI_Double_DQN.pptx")
OUTPUT = Path(__file__).with_name("Snake_AI_Double_DQN_revised.pptx")

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS = {"a": A_NS, "p": P_NS}


# (slide number, identifying text, corrected text)
REPLACEMENTS = [
    (2, "Agent phải tự học ăn táo", "Agent tự học chiến lược ăn táo, tránh va chạm và sống sót thông qua tương tác thử–sai. Luật chơi, trạng thái và reward được định nghĩa trong môi trường; chính sách chơi không được lập trình cứng."),
    (2, "Sparse rewards", "Delayed rewards"),
    (2, "Phần thưởng chỉ đến khi ăn được táo", "Một hành động có thể chỉ thể hiện hậu quả sau nhiều bước; reward shaping cung cấp tín hiệu hỗ trợ trong quá trình học"),
    (3, "Không cần công thức Bellman", "Agent học bằng cách giảm sai lệch giữa Q-value dự đoán và Double DQN target."),
    (3, "Lặp lại liên tục qua từng episode", "Double DQN target: y = r + γ Q_target(s′, argmax_a Q_online(s′,a))  ·  Loss: Huber(Q(s,a), y)"),
    (4, "Train cập nhật Q Network", "Transition được lưu vào Replay Buffer; Online Network được cập nhật trong mỗi episode. Target Network đồng bộ định kỳ sau 500 training steps."),
    (5, "Wall distance", "Wall distance (4)"),
    (5, "Snake length", "Snake length (1)"),
    (5, "Reachable free space", "Reachable space (3) + food distance (1)"),
    (6, "Hidden size mặc định", "Kiến trúc checkpoint: 28 → FC 256 → ReLU → FC 256 → ReLU → 3 Q-values (không dùng Softmax)"),
    (7, "Replay Buffer", "Replay Buffer (có ở DQN chuẩn)"),
    (7, "Target Network", "Target Network (có ở DQN chuẩn)"),
    (8, "Cùng một mạng vừa chọn", "DQN lấy max trực tiếp trên Target Network; Double DQN dùng Online Network để chọn action và Target Network để đánh giá action đó."),
    (8, "→ Overestimate", "y = r + γ maxₐ Q_target(s′,a)  → dễ overestimate"),
    (8, "target Q (ổn định hơn)", "y = r + γ Q_target(s′, argmaxₐ Q_online(s′,a))"),
    (9, "Long training is still in progress.", "ĐÃ HOÀN TẤT HUẤN LUYỆN 500 EPISODES"),
    (9, "TRẠNG THÁI HIỆN TẠI", "THIẾT LẬP ĐÁNH GIÁ"),
    (9, "Đã kiểm thử ngắn", "Checkpoint đánh giá: snake_dqn_500.pth  ·  Board: 20×20. Cần báo cáo kết quả trên nhiều episode với cùng seed và cấu hình; không dùng Accuracy cho bài toán RL."),
    (9, "Có thể bổ sung", "Bổ sung trước khi bảo vệ: mean/median/max apples · average reward · average episode length · learning curve · gameplay demo"),
    (10, "Dueling DQN", "Dueling DQN (cải tiến RL)"),
    (10, "Prioritized Replay", "Prioritized Replay (cải tiến RL)"),
    (10, "Rainbow DQN", "Rainbow DQN (cải tiến RL)"),
    (10, "Curriculum Learning", "Curriculum Learning (chiến lược train)"),
    (10, "Hamiltonian Strategy", "Hamiltonian / tail-reachability planning"),
]


def shape_text(shape):
    return "\n".join((node.text or "") for node in shape.findall(".//a:t", NS))


def replace_shape_text(shape, replacement):
    nodes = shape.findall(".//a:t", NS)
    if not nodes:
        return
    nodes[0].text = replacement
    for node in nodes[1:]:
        node.text = ""


def revise_slide(xml_data, slide_number):
    root = ET.fromstring(xml_data)
    pending = [(needle, text) for number, needle, text in REPLACEMENTS if number == slide_number]
    found = set()
    for shape in root.findall(".//p:sp", NS):
        current = shape_text(shape)
        # Baseline DQN chuẩn cũng sử dụng replay buffer và target network.
        if slide_number == 7 and current == "—":
            offset = shape.find(".//a:xfrm/a:off", NS)
            if offset is not None and offset.get("y") in {"2624328", "3191256"}:
                replace_shape_text(shape, "✓")
                continue
        for needle, replacement in pending:
            if needle in current:
                replace_shape_text(shape, replacement)
                found.add(needle)
                break
    missing = [needle for needle, _ in pending if needle not in found]
    if missing:
        print(f"Slide {slide_number}: not found: {missing}")
    ET.register_namespace("a", A_NS)
    ET.register_namespace("p", P_NS)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


with ZipFile(SOURCE, "r") as source, ZipFile(OUTPUT, "w", ZIP_DEFLATED) as target:
    for item in source.infolist():
        data = source.read(item.filename)
        if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
            number = int(item.filename.removeprefix("ppt/slides/slide").removesuffix(".xml"))
            data = revise_slide(data, number)
        target.writestr(item, data)

print(OUTPUT)
