# Sơ đồ thuật toán Multi-Objective Whale Optimizer

```mermaid
flowchart TD
    A["Bắt đầu"] --> B["Khởi tạo quần thể cá voi"]
    B --> C["Khởi tạo archive và grid"]
    C --> D["Bắt đầu vòng lặp chính"]
    D --> E["Cập nhật tham số a và a2"]
    E --> F["Cập nhật vị trí tất cả cá voi"]
    F --> G["Chọn lãnh đạo từ archive"]
    G --> H["Tính p ngẫu nhiên"]
    H --> I{"p < 0.5?"}
    I -- "Đúng" --> J{"|A| >= 1?"}
    I -- "Sai" --> K["Tấn công bằng bong bóng mạng"]
    J -- "Đúng" --> L["Tìm kiếm con mồi"]
    J -- "Sai" --> M["Vây quanh con mồi"]
    L --> N["Kiểm tra biên và cập nhật fitness"]
    M --> N
    K --> N
    N --> O["Cập nhật archive với quần thể hiện tại"]
    O --> P["Lưu trạng thái archive"]
    P --> Q{"iter >= max_iter?"}
    Q -- "Chưa" --> D
    Q -- "Rồi" --> R["Kết thúc"]
    
    subgraph Khởi tạo
    B
    C
    end
    
    subgraph Vòng lặp chính
    E
    F
    G
    H
    I
    J
    L
    M
    K
    N
    O
    P
    end
```

### Giải thích chi tiết các bước:

1. **Khởi tạo quần thể cá voi**: 
   - Tạo ngẫu nhiên các vị trí ban đầu trong không gian tìm kiếm
   - Mỗi vị trí X_i ∈ [lb, ub]^dim
   - Tính toán giá trị hàm mục tiêu đa mục tiêu

2. **Khởi tạo archive và grid**:
   - Xác định các giải pháp không bị chi phối từ quần thể ban đầu
   - Khởi tạo archive với các giải pháp không bị chi phối
   - Tạo grid để quản lý archive dựa trên không gian mục tiêu

3. **Vòng lặp chính** (max_iter lần):
   - **Cập nhật tham số a và a2**: 
     * Giảm tuyến tính theo số lần lặp
     ```python
     a = 2 - iter * (2 / max_iter)
     a2 = -1 + iter * ((-1) / max_iter)
     ```

   - **Chọn lãnh đạo từ archive**:
     * Sử dụng lựa chọn dựa trên grid để chọn lãnh đạo từ archive
     * Nếu archive rỗng, chọn ngẫu nhiên từ quần thể

   - **Cập nhật vị trí tất cả cá voi**:
     * Mỗi cá voi cập nhật vị trí dựa trên hành vi săn mồi

   - **Tính p ngẫu nhiên**: 
     * p ∈ [0, 1] để quyết định hành vi săn mồi

   - **Nếu p < 0.5 (Vây quanh hoặc Tìm kiếm)**:
     * **Nếu |A| >= 1**: Tìm kiếm con mồi (khám phá)
       * Chọn lãnh đạo ngẫu nhiên từ archive hoặc quần thể
       ```python
       if self.archive:
           rand_leader = np.random.choice(self.archive)
           D_X_rand = abs(C * rand_leader.position[j] - whale.position[j])
           new_position[j] = rand_leader.position[j] - A * D_X_rand
       ```
     * **Nếu |A| < 1**: Vây quanh con mồi (khai thác)
       ```python
       D_leader = abs(C * leader.position[j] - whale.position[j])
       new_position[j] = leader.position[j] - A * D_leader
       ```

   - **Nếu p >= 0.5 (Tấn công bằng bong bóng mạng)**:
     * Di chuyển xoắn ốc quanh con mồi
     ```python
     distance_to_leader = abs(leader.position[j] - whale.position[j])
     new_position[j] = distance_to_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader.position[j]
     ```

   - **Kiểm tra biên và cập nhật fitness**:
     * Đảm bảo vị trí nằm trong biên [lb, ub]
     * Tính toán lại giá trị hàm mục tiêu đa mục tiêu

   - **Cập nhật archive với quần thể hiện tại**:
     * Thêm các giải pháp không bị chi phối vào archive
     * Duy trì kích thước archive bằng cách loại bỏ các giải pháp dư thừa
     * Cập nhật grid để phản ánh archive mới

   - **Lưu trạng thái archive**:
     * Lưu trữ archive hiện tại cho lịch sử tối ưu hóa

4. **Kết thúc**:
   - Lưu trữ kết quả cuối cùng
   - Trả về archive (tập các giải pháp Pareto tối ưu) và lịch sử
