import numpy as np


# Local Correlation Search Function
def find_optimal_offset(
    combined_events, index_mask, initial_rect, max_steps=5, delta=1, verbose=False
):
    def crop_index_roi(frame, rect):
        x1, y1, x2, y2 = rect
        return frame[y1:y2, x1:x2]

    def compute_score(event_roi, mask_roi):
        if (
            event_roi.shape != mask_roi.shape
            or event_roi.shape[0] == 0
            or event_roi.shape[1] == 0
        ):
            if verbose:
                print(
                    f"Shape Error: Event ROI: {event_roi.shape}, Mask ROI: {mask_roi.shape}"
                )
            return 0

        and_roi = event_roi & mask_roi
        positive_overlap = np.count_nonzero(and_roi)

        # Penalize for event pixels outside the mask
        outside_mask_roi = event_roi & ~mask_roi
        negative_overlap = np.count_nonzero(outside_mask_roi)

        return positive_overlap - negative_overlap

    # Initial setup
    step_offset = [0, 0]
    best_num_on_pixels = 0
    current_rect = np.array(initial_rect)

    # Preprocessings
    index_mask = index_mask.astype(bool)  # Ensure mask is boolean

    # Local search loop
    for step in range(max_steps):
        improved = False
        step_offset = [0, 0]
        for dx, dy in [(0, delta), (delta, 0), (0, -delta), (-delta, 0)]:
            # Get new rect, crop ROIs and compute overlap
            new_rect = current_rect + np.array([dx, dy, dx, dy])
            event_roi = crop_index_roi(combined_events, new_rect)
            mask_roi = crop_index_roi(index_mask, initial_rect)
            num_on_pixels = compute_score(event_roi, mask_roi)

            # If improved, update
            if num_on_pixels > best_num_on_pixels:
                best_num_on_pixels = num_on_pixels
                updated_rect = new_rect.copy()
                step_offset = [dx, dy]
                improved = True

        if not improved:
            if verbose:
                print("No improvement found; stopping search.")
            break

        current_rect = updated_rect.copy()

        if verbose:
            print(
                f"Step {step}: step offset: {step_offset}, Num on pixels: {best_num_on_pixels}"
            )

    total_offset = (current_rect - initial_rect)[:2]
    if verbose:
        print(f"Total offset: {total_offset}, Num on pixels: {best_num_on_pixels}")

    return total_offset, best_num_on_pixels
