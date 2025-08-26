# Sơ đồ thuật toán Artificial Ecosystem Optimizer

```mermaid
flowchart TD
    A["Bắt đầu"] --> B["Khởi tạo quần thể"]
    B --> C["Sắp xếp quần thể và tìm giải pháp tốt nhất"]
    C --> D["Khởi tạo lịch sử tối ưu hóa"]
    D --> E["Bắt đầu vòng lặp chính"]
    E --> F["Giai đoạn sản xuất: tạo sinh vật mới"]
    F --> G["Giai đoạn tiêu thụ: cập nhật hành vi tiêu thụ"]
    G --> H["Giai đoạn phân hủy: cập nhật hành vi phân hủy"]
    H --> I["Đánh giá và cập nhật quần thể"]
    I --> J["Cập nhật giải pháp tốt nhất"]
    J --> K["Lưu trữ giải pháp tốt nhất hiện tại"]
    K --> L{"iter >= max_iter?"}
    L -- "Chưa" --> E
    L -- "Rồi" --> M["Lưu trữ kết quả cuối cùng"]
    M --> N["Kết thúc"]
    
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
    
    subgraph Giai đoạn sản xuất
    F1["Tạo vị trí ngẫu nhiên trong không gian tìm kiếm"]
    F2["Tính toán trọng số sản xuất a = (1 - iter/max_iter) * r1"]
    F3["Tạo sinh vật mới: (1-a)*best_position + a*random_position"]
    F --> F1 --> F2 --> F3
    end
    
    subgraph Giai đoạn tiêu thụ
    G1["Tạo hệ số tiêu thụ C sử dụng Levy flight"]
    G2{"r < 1/3?"}
    G2 -- "Có" --> G3["Tiêu thụ từ producer"]
    G2 -- "1/3 ≤ r < 2/3" --> G4["Tiêu thụ từ consumer ngẫu nhiên"]
    G2 -- "r ≥ 2/3" --> G5["Tiêu thụ từ cả producer và consumer"]
    G --> G1 --> G2
    end
    
    subgraph Giai đoạn phân hủy
    H1["Tạo hệ số phân hủy weight_factor = 3 * N(0,1)"]
    H2["Tính toán vị trí mới dựa trên best_solver"]
    H3["Cập nhật vị trí sử dụng phương trình phân hủy"]
    H --> H1 --> H2 --> H3
    end
```

### Giải thích chi tiết các bước:

1. **Khởi tạo quần thể**:
   - Tạo ngẫu nhiên các vị trí ban đầu trong không gian tìm kiếm
   - Mỗi sinh vật có vị trí và giá trị fitness
   - Tính toán giá trị hàm mục tiêu objective_func(position)

2. **Sắp xếp quần thể và tìm giải pháp tốt nhất**:
   - Sắp xếp quần thể theo thứ tự fitness
   - Chọn giải pháp tốt nhất làm best_solver

3. **Khởi tạo lịch sử tối ưu hóa**:
   - Khởi tạo danh sách lưu trữ lịch sử các giải pháp tốt nhất

4. **Vòng lặp chính** (max_iter lần):
   - **Giai đoạn sản xuất**:
     * Tạo sinh vật mới dựa trên vị trí tốt nhất và vị trí ngẫu nhiên
     * Trọng số sản xuất giảm dần theo số lần lặp
     ```python
     a = (1 - iter / max_iter) * r1
     new_position = (1 - a) * best_position + a * random_position
     ```

   - **Giai đoạn tiêu thụ**:
     * Các sinh vật cập nhật vị trí dựa trên hành vi tiêu thụ
     * Sử dụng hệ số tiêu thụ C tính bằng Levy flight
     ```python
     u = np.random.normal(0, 1, self.dim)
     v = np.random.normal(0, 1, self.dim)
     C = 0.5 * u / np.abs(v)
     ```
     * Ba chiến lược tiêu thụ:
       - Tiêu thụ từ producer (xác suất < 1/3)
       - Tiêu thụ từ consumer ngẫu nhiên (1/3 ≤ xác suất < 2/3)
       - Tiêu thụ từ cả producer và consumer (xác suất ≥ 2/3)

   - **Giai đoạn phân hủy**:
     * Các sinh vật cập nhật vị trí dựa trên hành vi phân hủy
     * Sử dụng hệ số phân hủy weight_factor
     ```python
     weight_factor = 3 * np.random.normal(0, 1)
     new_position = best_solver.position + weight_factor * (
         (r3 * random_multiplier - 1) * best_solver.position -
         (2 * r3 - 1) * population[i].position
     )
     ```

   - **Đánh giá và cập nhật quần thể**:
     * So sánh quần thể mới và quần thể cũ
     * Giữ lại các giải pháp tốt hơn

   - **Cập nhật giải pháp tốt nhất**:
     * So sánh và cập nhật nếu tìm thấy giải pháp tốt hơn

   - **Lưu trữ giải pháp tốt nhất hiện tại**:
     * Lưu trữ best_solver vào lịch sử

5. **Kết thúc**:
   - Lưu trữ kết quả cuối cùng
   - Hiển thị lịch sử tối ưu hóa
   - Trả về giải pháp tốt nhất và lịch sử
