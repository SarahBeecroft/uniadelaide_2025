import time
import torch

def main():
    # Use first visible GPU
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to PyTorch (HIP/ROCm).")

    device = torch.device("cuda:0")

    # Big-ish matrices – adjust size or dtype if you need more/less load
    size = 8192
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Do heavy matmuls for ~5 minutes
    duration_sec = 5 * 60
    start = time.time()
    it = 0
    while time.time() - start < duration_sec:
        # Matrix multiply and a bit of extra work to keep the ALUs busy
        c = a @ b
        c = torch.relu(c)
        # Optional: prevent compiler from optimizing everything away
        _ = c.mean()

        it += 1
        # Occasionally sync so work is actually issued, but not every loop
        if it % 10 == 0:
            torch.cuda.synchronize()

    # Final sync to ensure all kernels have finished before exiting
    torch.cuda.synchronize()
    print(f"Done. Iterations: {it}")

if __name__ == "__main__":
    main()
