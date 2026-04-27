import torch

if __name__ == "__main__":
    print(f"{torch.tensor(0.01, dtype=torch.float16):.10f}")
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s) # expect [10]
    
    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s) # expect [10]
    
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s) # expect [10]
    
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s) # expect [10]
    
    a = torch.tensor(1.0, dtype=torch.float16)
    b = a + torch.tensor(0.01, dtype=torch.float16)
    print(f"{b:.6f}")