import torch

print('-'*30)
print('Now start the program')
print('-'*30)
def get_device():
    """
        Check if CUDA or mac m1 GPU is available
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
print('-'*30)

def main():
    print('main')

if __name__ == '__main__':
    main()
