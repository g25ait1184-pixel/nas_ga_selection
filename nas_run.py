import torch, sys, os, pickle
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from model_ga import GeneticAlgorithm
from model_cnn import CNN

# if __name__ == "__main__":

parent = os.path.abspath('')
if not os.path.exists(os.path.join(parent, 'outputs')):
    os.mkdir(os.path.join(parent, 'outputs'))
all_logs = [i for i in os.listdir(os.path.join(parent, 'outputs')) if 'log' in i]
os.mkdir(os.path.join(parent, 'outputs', f'run_{len(all_logs)+1}'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.stdout = open(os.path.join(parent, 'outputs', f'run_{len(all_logs)+1}', f'nas_run.log'), 'w')

print(f"Using device: {device}", flush=True)

# Load CIFAR-10 dataset (reduced for faster NAS)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
valset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use only 5000 samples for quick NAS
train_subset = Subset(trainset, range(5000))
val_subset = Subset(valset, range(1000))

train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

# Run NAS with GA
def run_ga_experiment(selection_method, run_id):

    run_dir, log_file = start_run_logger(run_id)

    print("===============================================")
    print(f"   STARTING RUN {run_id} — {selection_method}")
    print("===============================================\n")

    ga = GeneticAlgorithm(
        population_size=10,
        generations=5,
        mutation_rate=0.3,
        crossover_rate=0.7
    )

    best_arch = ga.evolve(train_loader, val_loader, device, run=run_id,
                          selection_method=selection_method)

    print("\n======================")
    print("FINAL BEST ARCHITECTURE")
    print("======================")
    print(f"Genes:     {best_arch.genes}")
    print(f"Accuracy:  {best_arch.accuracy:.4f}")
    print(f"Fitness:   {best_arch.fitness:.4f}")

    final_model = CNN(best_arch.genes).to(device)
    print(f"\nTotal parameters: {sum(p.numel() for p in final_model.parameters()):,}")
    print("\nModel architecture:\n", final_model)

    with open(os.path.join(run_dir, "best_arch.pkl"), "wb") as f:
        pickle.dump(best_arch, f)

    # close log file
    log_file.close()

    # restore stdout
    sys.stdout = sys.__stdout__

    print(f"Finished run {run_id} ({selection_method}). Log saved in: {run_dir}/nas_run.log\n")

    return best_arch



# -------------------------------------------------------------------
# Run BOTH methods: Tournament + Roulette Wheel
# -------------------------------------------------------------------
run1 = run_ga_experiment("tournament", run_id=1)
run2 = run_ga_experiment("roulette",   run_id=2)

print("\n===== ALL RUNS COMPLETED =====")
print("Tournament fitness:", run1.fitness)
print("Roulette fitness:", run2.fitness)
print("Logs saved under outputs/run_*/nas_run.log")

with open(os.path.join(parent, 'outputs', f'run_{len(all_logs)+1}', f"best_arch.pkl"), 'wb') as f:
    pickle.dump(best_arch, f)

sys.stdout = sys.__stdout__