/*
 * Per-sample FxLMS adaptive filter core for the ANC queue runner.
 *
 * This is the part of OnlineFxLMSFilter.process_sample that ran ~48,000
 * times/sec in Python (once per audio sample), each time paying numpy
 * dispatch overhead on tiny (16-50 element) arrays. Moved here it's a
 * tight loop over plain float buffers with no per-sample allocation.
 *
 * All persistent buffers (control_weights, histories, secondary path,
 * delay buffer) are owned by numpy on the Python side and handed in as
 * raw pointers via ctypes -- this struct just describes that memory.
 */

typedef struct {
    int control_length;
    int secondary_length;
    int secondary_start;
    int delay_length;
    int delay_index;

    float learning_rate;
    float epsilon;
    float leakage;
    float update_sign;
    float max_command;

    float *control_weights;
    float *reference_history;
    float *filtered_reference_history;
    float *secondary_path;
    float *secondary_history;
    float *delay_buffer;
} FxLMSState;


static inline float clipf(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}


void fxlms_reset(FxLMSState *state) {
    for (int i = 0; i < state->control_length; i++) {
        state->control_weights[i] = 0.0f;
        state->reference_history[i] = 0.0f;
        state->filtered_reference_history[i] = 0.0f;
    }
    for (int i = 0; i < state->secondary_length; i++) {
        state->secondary_history[i] = 0.0f;
    }
    for (int i = 0; i < state->delay_length; i++) {
        state->delay_buffer[i] = 0.0f;
    }
    state->delay_index = 0;
}


void fxlms_process_block(
    FxLMSState *state,
    const float *reference_block,
    const float *error_block,
    float *command_block,
    int block_length,
    int adapt
) {
    const int control_length = state->control_length;
    const int secondary_length = state->secondary_length;
    const int secondary_start = state->secondary_start;

    float * const control_weights = state->control_weights;
    float * const reference_history = state->reference_history;
    float * const filtered_reference_history = state->filtered_reference_history;
    const float * const secondary_path = state->secondary_path;
    float * const secondary_history = state->secondary_history;
    float * const delay_buffer = state->delay_buffer;

    const float learning_rate = state->learning_rate;
    const float epsilon = state->epsilon;
    const float one_minus_leakage = 1.0f - state->leakage;
    const float update_sign = state->update_sign;
    const float max_command = state->max_command;

    int delay_index = state->delay_index;

    for (int n = 0; n < block_length; n++) {
        const float reference_sample = reference_block[n];
        const float error_sample = error_block[n];

        /* Push reference_sample to the front of reference_history (FIFO shift) */
        for (int i = control_length - 1; i > 0; i--) {
            reference_history[i] = reference_history[i - 1];
        }
        reference_history[0] = reference_sample;

        /* command = clip(dot(control_weights, reference_history)) */
        float command = 0.0f;
        for (int i = 0; i < control_length; i++) {
            command += control_weights[i] * reference_history[i];
        }
        command_block[n] = clipf(command, -max_command, max_command);

        /* Delay-aware secondary path filtering via circular delay buffer */
        float delayed_reference;
        if (secondary_start > 0) {
            delayed_reference = delay_buffer[delay_index];
            delay_buffer[delay_index] = reference_sample;
            delay_index++;
            if (delay_index >= secondary_start) {
                delay_index = 0;
            }
        } else {
            delayed_reference = reference_sample;
        }

        for (int i = secondary_length - 1; i > 0; i--) {
            secondary_history[i] = secondary_history[i - 1];
        }
        secondary_history[0] = delayed_reference;

        float filtered_reference_sample = 0.0f;
        for (int i = 0; i < secondary_length; i++) {
            filtered_reference_sample += secondary_path[i] * secondary_history[i];
        }

        for (int i = control_length - 1; i > 0; i--) {
            filtered_reference_history[i] = filtered_reference_history[i - 1];
        }
        filtered_reference_history[0] = filtered_reference_sample;

        if (adapt) {
            float filtered_power = 0.0f;
            for (int i = 0; i < control_length; i++) {
                filtered_power += filtered_reference_history[i] * filtered_reference_history[i];
            }

            const float scale = (learning_rate * error_sample) / (filtered_power + epsilon);

            for (int i = 0; i < control_length; i++) {
                control_weights[i] = one_minus_leakage * control_weights[i]
                    + update_sign * (scale * filtered_reference_history[i]);
            }
        }
    }

    state->delay_index = delay_index;
}
