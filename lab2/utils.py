import matplotlib.pyplot as plt

def extract_numerical_answer(ans: str) -> str:
    ind = ans.rindex('\n') + 1
    ans_marker = '#### '
    if ans[ind: ind + len(ans_marker)] != ans_marker:
        raise ValueError(
            f'Incorrectly formatted answer `{ans}` does not have '
            f'`{ans_marker}` at the end'
        )
    return ans[ind + len(ans_marker):]

def plot_train_vs_val(log_history):
    # Separate training and evaluation logs
    train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    # Extract values
    train_steps = [log['step'] for log in train_logs]
    train_losses = [log['loss'] for log in train_logs]

    eval_steps = [log['step'] for log in eval_logs]
    eval_losses = [log['eval_loss'] for log in eval_logs]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Training Loss', marker='o')
    plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='s')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # If you want to see the actual data
    print("Training losses:", train_losses)
    print("Validation losses:", eval_losses)