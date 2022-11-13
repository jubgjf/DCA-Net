import config
import random

random.seed(8)

if __name__ == '__main__':
    labels: list
    data: list
    lens: int

    # Read labels
    with open(f'../{config.data_path}/intent_label.txt') as f:
        labels_raw = f.readlines()
        labels = [line.strip().split('\t')[1] for line in labels_raw]

    # Read training dataset
    with open(f'../{config.data_path}/train_clean.txt') as f:
        data_raw = f.readlines()
        data = [d.strip() for d in data_raw]
        lens = len(data)

    # Probabilities of incorrect feedback of users
    # `user_noise_p[i]` -> user[i] mistake probability
    user_noise_p = [random.gauss(config.user_mean, config.user_std) for _ in range(config.user_count)]

    # Add noise
    # random num < user mistake prob -> make mistake
    # random num > user mistake prob -> correct
    for i in range(lens):
        current_user_index = int(i / (lens / config.user_count))
        if data[i] == '':
            assert data[i - 1] in labels
            if random.random() < user_noise_p[current_user_index]:
                data[i - 1] = random.choice(labels)

    # train     -> noisy
    # dev, test -> clean
    with open(f'../{config.data_path}/train.txt', 'w') as f:
        noisy_data = [d + '\n' for d in data]
        f.writelines(noisy_data)
