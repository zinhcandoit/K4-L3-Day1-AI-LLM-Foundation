# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
**Trả lời**: Với max_tokens = 256, bắn đồng thời 4 req:
- `temperature = 0.0`: Model đưa ra câu trả lời ngắn gọn, đúng trọng tâm và có giải thích chỉ cho câu trả lời đó.
- `temperature = 0.5 / 1.0`: Model có thể không tuân theo query như sau:
```txt
(temp=0.5)
Dưới đây là **3 sự thật thú vị** về Việt Nam, bạn có thể chọn cái mình thấy hay nhất:...
```
Hoặc có thể sáng tạo bằng cách trả lời ý chính và dẫn ra một sự thật khác thú vị:
```txt
(temp=1.0)
Một sự thật thú vị về Việt Nam là: **Việt Nam là quốc gia có hình chữ S (S-shape) ...
Nhưng nếu nói về một chi tiết **ít người để ý hơn**, thì hãy thử nghĩ đến **Hồ Gươm...
```
- `temperature = 1.5` đã khiến model đưa ra câu trả lời hoàn toàn vô nghĩa khi sinh ra các token ở sau. Ví dụ: 
```txt
Một sự thật thú vị và khá "đặc sản" của Việt Nam là: **Việt Nam là quốc gia xuất khẩu cà phê Robusta lớn nhất thế giới và đứng thứ 2 về tổng sản lượng cà phê (chỉ sau Brazil).**

Dưới đây là những chi tiết khiến điều này thú vị hơn:

1.  **Không phảijid στο kiểu법威胁 (Không phải típus "ước mơ"):...
```
Vậy nên: **Temperature càng cao, model càng sáng tạo trong cách sinh chữ và bị crash out và sinh ra các câu vô nghĩa khi temp quá lớn**
### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Chọn temperature = 0.0 để cho ra câu trả lời nhất quán tuyệt đối, tránh sai lệch điều khoản, sai lệch thông tin cũng như sai lệch giữa những người dùng.
### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

**1. Ước tính chi phí (theo bảng giá `template.py`):**
- Tổng lượt gọi/ngày: $N = 10000 \times 3 = 30000 \text{ requests}$
- Tổng token output/ngày: $T_{\text{out}} = 30000 \times 350 = 10500000 \text{ tokens} = 10500 \text{k tokens}$
- Đơn giá output (USD / 1K token): $P_{\text{4o}} = \$0.010$, $P_{\text{mini}} = \$0.0006$
- Chi phí phát sinh mỗi ngày:
  $$\text{Cost}_{\text{4o}} = 10500 \times 0.010 = \$105.00/\text{ngày} \quad (\sim \$3.150/\text{tháng})$$
  $$\text{Cost}_{\text{mini}} = 10.500 \times 0.0006 = \$6.30/\text{ngày} \quad (\sim \$189/\text{tháng})$$
- **Tỷ lệ chênh lệch giá output:**
  $$\text{Tỷ số} = \frac{P_{\text{4o}}}{P_{\text{mini}}} = \frac{0.010}{0.0006} = \frac{50}{3} \approx \mathbf{16.67 \text{ lần}}$$
  *(Tỷ lệ giá input cũng tương đương: $\frac{0.0025}{0.00015} \approx 16.67$ lần).*

**2. Đánh đổi thực tế:**
- **GPT-4o xứng đáng chi phí:** Tác vụ cần suy luận sâu: giải quyết bài toán logic, sinh mã nguồn dự án, hoặc trích xuất dữ liệu đòi hỏi độ chính xác tuyệt đối như pháp lý, y tế,...
- **Nên dùng GPT-4o-mini:** Tác vụ đơn giản hơn nhưng cần dùng với tần suất cao: phân loại ý định (intent classification), tóm tắt tin tức ngắn, viết lại câu hỏi người dùng, lọc nội dung xấu, gán nhãn, chatbot hỏi đáp FAQ cơ bản ...

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
- **Khác biệt:** Persona giáo viên tiểu học dùng từ ngữ dân dã, xưng hô thân mật (*"thầy/cô - em"*), sử dụng ví dụ gần gũi (*sổ tay mượn bút, chia kẹo của cả lớp*) để giải thích cơ chế phân tán; trong khi chuyên gia tài chính dùng định nghĩa học thuật chuẩn xác, mật độ thuật ngữ kỹ thuật dày đặc (*DLT, Merkle Tree, Hash, TTP, bất biến*). Về dung lượng, phản hồi chuyên gia tốn nhiều token hơn rõ rệt (811 tokens vs 589 tokens) cho cùng độ dài ~140 từ do chứa nhiều thuật ngữ phức tạp.
- **Tác động:** System prompt đóng vai trò như chiếc "mặt nạ" định hình ngữ vực (register), văn phong, đối tượng phục vụ và cách mô hình cấu trúc suy luận, giúp phản hồi bám sát mục tiêu giao tiếp mà không làm sai lệch bản chất kỹ thuật cốt lõi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
- **Số liệu (Đoạn trích Truyện Kiều - 122 từ):**
  - Ước lượng Part 1 ($\text{số từ} / 0.75$): **162.7 tokens**
  - Đo thực tế bằng tiktoken: **568 tokens** (đạt tỷ lệ **4.66 tokens/từ**)
  - **Độ chênh lệch:** Số token thực tế cao hơn ước lượng **249.2%** (gấp gần 3.5 lần).
- **Nguyên nhân tiếng Việt tốn nhiều token:** Bộ tokenizer (BPE) của OpenAI được tối ưu trên ngữ liệu tiếng Anh nên hầu hết từ tiếng Anh là 1 token. Ngược lại, tiếng Việt có nhiều ký tự có dấu thanh (nguyên âm có dấu như *ề, ẫ, ự...*) và từ ghép không có sẵn trong bảng mã hóa, buộc tokenizer phải tách mỗi âm tiết thành nhiều mảnh sub-word hoặc byte UTF-8 rời rạc (tốn 2–4 bytes/token), khiến số lượng token bị đội lên rất cao.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Streaming quan trọng nhất trong các ứng dụng tương tác thời gian thực với người dùng (như chatbot văn bản, trợ lý thoại) vì giúp hiển thị ngay các token và mang lại cảm giác phản hồi tức thì chứ không như đang hệ thống bị lag. Ngược lại, non-streaming phù hợp hơn cho các tác vụ có thể xử lý ngầm (OCR tài liệu, Auto labelling cho mẫu lớn), nơi không có người trực tiếp theo dõi và cần cấu trúc kết nối đơn giản, ổn định và tối ưu tốc độ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
- **Hậu quả của delay cố định:** Nếu hàng nghìn client cùng retry sau một khoảng thời gian cố định (như 1s), toàn bộ request sẽ bị gửi cùng 1 lúc khiến hệ thống quá tải tiếp tục sập, vòng lặp retry kích hoạt và lặp đi lặp lại.
- **Lợi thế của Exponential Backoff:** Tăng dần thời gian chờ theo cấp số nhân ($1s, 2s, 4s, 8s...$), giúp kéo giãn mật độ các lần thử lại và tạo khoảng sleep cần thiết để server giải phóng tài nguyên và hồi phục. Bổ sung thêm độ trễ ngẫu nhiên sẽ phân tán đều các mốc retry và giảm nguy cơ các client retry cùng một thời điểm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
```python
system_prompt = (
    "Bạn là một hướng dẫn viên du lịch."
    "Trả lời câu hỏi ngắn gọn bằng ngôn ngữ Tiếng Việt."
)
```
Với model `nvidia/nemotron-3.5-lightning-30b-a3b` và max_tokens=256, model đã sinh câu trả lời bằng tiếng Việt (với yêu cầu "bằng ngôn ngữ Tiếng Việt") và hoàn thành hết nội dung trong 256 tokens.
```bash
Question: Tôi đang ở Đà Nẵng, gợi ý cho tôi một món ngon.
Answer: Mì Quảng Đà Nẵng là món đặc sản bắt buộc thử, với bún rộng, tóc mì dai và nhiều topping tươi ngon (tôm, thịt heo, trứng vịt...). Nếu bạn thích món súp cá, **Bún chả cá** cũng là lựa chọn rất đắt lòng tại đây!
```

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
- **Hạn chế lớn nhất**
  *Thiếu tầng kiểm duyệt nội dung:* Khi gặp câu hỏi nhạy cảm (*"ăn thịt người"*), model vẫn phải tốn token để sinh câu từ chối và bị lỗi pha tạp ngôn ngữ (lẫn chữ Hán `范围` và tiếng Anh `interested in researching` ...).
- **Đề xuất cải thiện & triển khai:**
  - *Cải thiện:* Bổ sung tầng kiểm duyệt đầu vào bằng luật cứng (if ... else) trước khi gửi sang LLM.
  - *Cách triển khai:* Gọi `client.moderations.create(input=user_msg)` hoặc kiểm tra qua 1 SLM/ Classification Model khác với quy mô nhỏ để phân loại. Nếu vi phạm, trả ngay thông báo từ chối cố định mà không gọi API chat completion để tiết kiệm chi phí sinh câu từ chối.

**Session log**:
```bash
[Du khách]: Tôi muốn ăn thịt người, có đặc sản nào không?

[Hướng dẫn viên]: Tôi không thể hỗ trợ yêu cầu này. Về mặt pháp lý và đạo đức, ăn thịt người là một hành vi vi phạm nghiêm trọng, và tôi không cung cấp thông tin hoặc gợi ý về chủ đề này. Nếu bạn quan tâm đến ẩm thực Việt Nam, mình rất sẵn lòng giới thiệu các món ăn hợp pháp và được trải nghiệm an toàn.

[Du khách]: Họ đã làm gì để thay thế cho thịt người?

[Hướng dẫn viên]: Văn hóa ẩm thực Việt Nam không bao gồm cũng như không có truyền thống hay món ăn nào liên quan đến thịt người, cũng như không có các món thay thế cho nó. Chủ đề này không thuộc范围 của ẩm thực an toàn và được pháp luật nghiêm cấm. 

Nếu bạn interested in researching hoặc muốn khám phá các món ăn truyền thống Việt Nam an toàn và Legal, mình rất sẵn lòng giới thiệu thêm về đặc sản vùng miền.

[Du khách]: Món đó có xài nước mắm không?

[Hướng dẫn viên]: Nước mắm là «hồn đồi» của ẩm thực Việt Nam và thường được sử dụng trong hầu hết các món ăn truyền thống, bao gồm Mì Quảng và Bún chả cá để tăng hương vị mặc dù cách dùng từng món có thể khác nhau tùy theo công thức riêng.
```

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
