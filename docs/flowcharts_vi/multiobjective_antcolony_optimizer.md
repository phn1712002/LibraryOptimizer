# Sơ đồ thuật toán Multi-Objective Ant Colony Optimizer

```mermaid
flowchart TD
    A["Bắt đầu"] --> B["Khởi tạo quần thể kiến"]
    B --> C["Khởi tạo archive và grid"]
    C --> D["Tính toán trọng số và xác suất lựa chọn"]
    D --> E["Bắt đầu vòng lặp chính"]
    E --> F["Tính toán means từ quần thể"]
    F --> G["Tính toán standard deviations"]
    G --> H["Lấy mẫu quần thể mới từ phân phối Gaussian"]
    H --> I["Cập nhật archive với quần thể mới"]
    I --> J["Chọn quần thể đa dạng từ archive"]
    J --> K["Lưu trạng thái archive"]
    K --> L{"iter >= max_iter?"}
    L -- "Chưa" --> E
    L -- "Rồi" --> M["Kết thúc"]
    
    subgraph Khởi tạo
    B
    C
    D
    end
    
    subgraph Vòng lặp chính
    F
    G
    H
    I
    J
    K
    end
```

### Giải thích chi tiết các bước:

1. **Khởi tạo quần thể kiến**: 
   - Tạo ngẫu nhiên các vị trí ban đầu trong không gian tìm kiếm
   - Mỗi vị trí X_i ∈ [lb, ub]^dim
   - Tính toán giá trị hàm mục tiêu đa mục tiêu

2. **Khởi tạo archive và grid**:
   - Xác định các giải pháp không bị chi phối từ quần thể ban đầu
   - Khởi tạo archive với các giải pháp không bị chi phối
   - Tạo grid để quản lý archive dựa trên không gian mục tiêu

3. **Tính toán trọng số và xác suất lựa chọn**:
   - Tính trọng số Gaussian kernel cho từng giải pháp
   ```python
   w = (1 / (np.sqrt(2 * np.pi) * self.q * n_pop)) * 
        np.exp(-0.5 * (((np.arange(n_pop)) / (self.q * n_pop)) ** 2))
   ```
   - Tính xác suất lựa chọn
   ```python
   p = w / np.sum(w)
   ```

4. **Vòng lặp chính** (max_iter lần):
   - **Tính toán means từ quần thể**:
     * Lấy vị trí của tất cả các giải pháp trong quần thể hiện tại
     ```python
     means = np.array([member.position for member in population])
     ```

   - **Tính toán standard deviations**:
     * Tính độ lệch chuẩn cho từng giải pháp dựa trên khoảng cách trung bình
     ```python
     for l in range(n_pop):
         D = np.sum(np.abs(means[l] - means), axis=0)
         sigma[l] = self.zeta * D / (n_pop - 1)
     ```

   - **Lấy mẫu quần thể mới từ phân phối Gaussian**:
     * Tạo giải pháp mới bằng cách lấy mẫu từ phân phối Gaussian
     * Với mỗi thành phần của giải pháp:
       - Chọn kernel Gaussian bằng roulette wheel selection
       - Tạo biến ngẫu nhiên Gaussian
       ```python
       l = roulette_wheel_selection(probabilities)
       new_position[i] = means[l, i] + sigma[l, i] * np.random.randn()
       ```
     * Kiểm tra biên và đánh giá fitness đa mục tiêu

   - **Cập nhật archive với quần thể mới**:
     * Thêm các giải pháp không bị chi phối vào archive
     * Duy trì kích thước archive bằng cách loại bỏ các giải pháp dư thừa
     * Cập nhật grid để phản ánh archive mới

   - **Chọn quần thể đa dạng từ archive**:
     * Sử dụng lựa chọn dựa trên grid để chọn quần thể đa dạng
     * Ưu tiên các ô grid có mật độ thấp để duy trì đa dạng
     * Nếu archive nhỏ hơn kích thước quần thể, bổ sung giải pháp ngẫu nhiên

   - **Lưu trạng thái archive**:
     * Lưu trữ archive hiện tại cho lịch sử tối ưu hóa

5. **Kết thúc**:
   - Lưu trữ kết quả cuối cùng
   - Trả về archive (tập các giải pháp Pareto tối ưu) và lịch sử
