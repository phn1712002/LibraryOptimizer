# Sơ đồ thuật toán Whale Optimization Algorithm

```mermaid
flowchart TD
    A["Bắt đầu"] --> B["Khởi tạo quần thể cá voi"]
    B --> C["Khởi tạo biến lưu trữ và leader"]
    C --> D["Bắt đầu vòng lặp chính"]
    D --> E["Cập nhật tham số a và a2"]
    E --> F["Cập nhật vị trí tất cả cá voi"]
    F --> G["Tính toán p ngẫu nhiên"]
    G --> H{"p < 0.5?"}
    H -- "Có" --> I{"|A| >= 1?"}
    H -- "Không" --> J["Tấn công bằng bong bóng (spiral)"]
    I -- "Có" --> K["Tìm kiếm con mồi (exploration)"]
    I -- "Không" --> L["Vây bắt con mồi (exploitation)"]
    K --> M["Kiểm tra biên và cập nhật fitness"]
    L --> M
    J --> M
    M --> N["Cập nhật giải pháp tốt nhất"]
    N --> O{"iter >= max_iter?"}
    O -- "Chưa" --> D
    O -- "Rồi" --> P["Kết thúc"]
    
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
    K
    L
    J
    M
    N
    end
```

## Giải thích chi tiết các bước:

1. **Khởi tạo quần thể cá voi**: 
   - Tạo ngẫu nhiên các vị trí ban đầu của cá voi trong không gian tìm kiếm
   - Mỗi vị trí X_i ∈ [lb, ub]^dim
   - Tính toán giá trị hàm mục tiêu objective_func(X_i)

2. **Khởi tạo biến lưu trữ và leader**:
   - Khởi tạo lịch sử tối ưu hóa
   - Tìm cá voi có fitness tốt nhất làm leader ban đầu
   - Khởi tạo giải pháp tốt nhất ban đầu

3. **Vòng lặp chính** (max_iter lần):
   - **Cập nhật tham số a và a2**:
     * Giảm tuyến tính theo số lần lặp
     ```python
     a = 2 - iter * (2 / max_iter)
     a2 = -1 + iter * ((-1) / max_iter)
     ```

   - **Cập nhật vị trí tất cả cá voi**:
     * Mỗi cá voi cập nhật vị trí dựa trên hành vi của cá voi thực

   - **Tính toán p ngẫu nhiên**:
     * p ∈ [0, 1] quyết định chiến lược di chuyển

   - **Nếu p < 0.5 (Vây bắt hoặc Tìm kiếm)**:
     * **Nếu |A| >= 1**: Tìm kiếm con mồi (exploration phase)
       ```python
       # Chọn cá voi ngẫu nhiên
       rand_leader_index = np.random.randint(0, search_agents_no)
       X_rand = population[rand_leader_index].position
       D_X_rand = abs(C * X_rand[j] - member.position[j])
       new_position[j] = X_rand[j] - A * D_X_rand
       ```
     * **Nếu |A| < 1**: Vây bắt con mồi (exploitation phase)
       ```python
       D_leader = abs(C * leader.position[j] - member.position[j])
       new_position[j] = leader.position[j] - A * D_leader
       ```

   - **Nếu p >= 0.5 (Tấn công bằng bong bóng)**:
     * Di chuyển theo hình xoắn ốc về phía leader
     ```python
     distance_to_leader = abs(leader.position[j] - member.position[j])
     new_position[j] = distance_to_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader.position[j]
     ```

   - **Kiểm tra biên và cập nhật fitness**:
     * Đảm bảo vị trí nằm trong biên [lb, ub]
     * Tính toán lại giá trị hàm mục tiêu

   - **Cập nhật giải pháp tốt nhất**:
     * So sánh và cập nhật nếu tìm thấy giải pháp tốt hơn
     * Cập nhật leader nếu cần

4. **Kết thúc**:
   - Lưu trữ kết quả cuối cùng
   - Hiển thị lịch sử tối ưu hóa
   - Trả về giải pháp tốt nhất và lịch sử

## Đặc điểm thuật toán WAO:
- **Mô phỏng hành vi săn mồi của cá voi**: Vây bắt, tấn công bằng bong bóng, tìm kiếm
- **Cân bằng exploration và exploitation**: Thông qua tham số A và p
- **Tham số điều chỉnh**: a giảm tuyến tính từ 2 về 0, điều khiển hành vi
- **Hình xoắn ốc**: Mô phỏng bong bóng của cá voi khi tấn công con mồi
