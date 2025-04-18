import os, torch
os.makedirs('data/signals', exist_ok=True)
for i in range(10):
    features = torch.randn(30, 10)               # seq_len=30, num_features=10
    label    = torch.randint(0, 3, ()).item()    # random label
    torch.save({'features': features, 'label': label},
               f'data/signals/sample_{i}.pt')
print('Dummy signals generated.')
