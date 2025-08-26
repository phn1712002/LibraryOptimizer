# Sơ đồ thuật toán Moss Growth Optimizer

```mermaid
flowchart TD
    A["Bắt đầu"] --> B["Khởi tạo quần thể rêu"]
    B --> C["Sắp xếp quần thể và khởi tạo giải pháp tốt nhất"]
    C --> D["Khởi tạo cơ chế cryptobiosis"]
    D --> E["Bắt đầu vòng lặp chính"]
    E --> F["Tính toán tỷ lệ tiến trình"]
    F --> G["Ghi nhận thế hệ đầu cho cryptobiosis"]
    G --> H["Xử lý từng tác nhân tìm kiếm"]
    H --> I["Chọn vị trí tính toán dựa trên vùng đa số"]
    I --> J["Chia quần thể và chọn vùng có nhiều cá thể"]
    J --> K["Tính toán khoảng cách từ cá thể đến tốt nhất"]
    K --> L["Tính toán hướng gió (trung bình khoảng cách)"]
    L --> M["Tính toán tham số beta và gamma"]
    M --> N["Tính toán kích thước bước di chuyển"]
    N --> O{"Xác suất phân tán bào tử > d1?"}
    O -- "Có" --> P["Tìm kiếm phân tán bào tử: position + step2 * D_wind"]
    O -- "Không" --> Q["Tìm kiếm phân tán bào tử: position + step * D_wind"]
    P --> R{"Xác suất lan truyền kép < 0.8?"}
    Q --> R
    R -- "Có" --> S{"Xác suất > 0.5?"}
    R -- "Không" --> T["Kiểm tra biên và đánh giá fitness"]
    S -- "Có" --> U["Cập nhật chiều cụ thể: best + step3 * D_wind[dim]"]
    S -- "Không" --> V["Cập nhật tất cả chiều với hàm kích hoạt: (1-act)*new + act*best"]
    U --> T
    V --> T
    T --> W["Cập nhật quần thể và ghi nhận cryptobiosis"]
    W --> X{"rec >= rec_num hoặc iter cuối?"}
    X -- "Có" --> Y["Cơ chế cryptobiosis: khôi phục vị trí tốt nhất lịch sử"]
    X -- "Không" --> Z["Tăng bộ đếm rec"]
    Y --> AA["Cập nhật giải pháp tốt nhất"]
    Z --> AA
    AA --> AB["Lưu trữ giải pháp tốt nhất"]
    AB --> AC{"iter >= max_iter?"}
    AC -- "Chưa" --> E
    AC -- "Rồi" --> AD["Kết thúc"]
    
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
    L
    M
    N
    O
    P
    Q
    R
    S
    T
    U
    V
    W
    X
    Y
    Z
    AA
    AB
    end
```

### Giải thích chi tiết các bước:

1. **Khởi tạo quần thể rêu**: 
   - Tạo ngẫu nhiên các vị trí ban đầu trong không gian tìm kiếm
   - Mỗi vị trí X_i ∈ [lb, ub]^dim
   - Tính toán giá trị hàm mục tiêu objective_func(X_i)

2. **Sắp xếp quần thể và khởi tạo giải pháp tốt nhất**:
   - Sắp xếp quần thể dựa trên giá trị fitness
   - Chọn giải pháp tốt nhất ban đầu

3. **Khởi tạo cơ chế cryptobiosis**:
   - Khởi tạo bộ nhớ để ghi nhận lịch sử vị trí và fitness
   - rM: Lưu trữ lịch sử vị trí
   - rM_cos: Lưu trữ lịch sử fitness

4. **Vòng lặp chính** (max_iter lần):
   - **Tính toán tỷ lệ tiến trình**:
     ```python
     progress_ratio = current_fes / max_fes
     ```

   - **Ghi nhận thế hệ đầu cho cryptobiosis**:
     * Lưu trữ vị trí và fitness ban đầu

   - **Chọn vị trí tính toán dựa trên vùng đa số**:
     * Chia không gian tìm kiếm và chọn vùng có nhiều cá thể

   - **Tính toán khoảng cách và hướng gió**:
     ```python
     D = best_solution.position - cal_positions
     D_wind = np.mean(D, axis=0)
     ```

   - **Tính toán tham số beta và gamma**:
     ```python
     beta = cal_positions.shape[0] / search_agents_no
     gamma = 1 / np.sqrt(1 - beta**2) if beta < 1 else 1.0
     ```

   - **Tìm kiếm phân tán bào tử**:
     * **Nếu xác suất > d1**: Sử dụng step2
       ```python
       new_position = population[i].position + step2 * D_wind
       ```
     * **Nếu xác suất <= d1**: Sử dụng step
       ```python
       new_position = population[i].position + step * D_wind
       ```

   - **Tìm kiếm lan truyền kép**:
     * **Cập nhật chiều cụ thể**:
       ```python
       new_position[dim_idx] = best_solution.position[dim_idx] + step3 * D_wind[dim_idx]
       ```
     * **Cập nhật tất cả chiều với hàm kích hoạt**:
       ```python
       new_position = (1 - act) * new_position + act * best_solution.position
       ```

   - **Kiểm tra biên và đánh giá fitness**:
     * Đảm bảo vị trí nằm trong biên [lb, ub]
     * Tính toán giá trị hàm mục tiêu cho vị trí mới

   - **Cơ chế cryptobiosis**:
     * Khi đủ bản ghi hoặc iteration cuối, khôi phục vị trí tốt nhất lịch sử
     * Đặt lại bộ đếm rec

   - **Cập nhật giải pháp tốt nhất**:
     * So sánh và cập nhật nếu tìm thấy giải pháp tốt hơn

   - **Lưu trữ giải pháp tốt nhất**:
     * Lưu lại giải pháp tốt nhất tại mỗi iteration

5. **Kết thúc**:
   - Lưu trữ kết quả cuối cùng
   - Hiển thị lịch sử tối ưu hóa
   - Trả về giải pháp tốt nhất và lịch sử
