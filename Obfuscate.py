from Utils import *
from Replacer import *
from NN import *
from datetime import datetime
import pandas as pd
from torch.utils.data import DataLoader


def clean(texts):
    ret = []

    for elem in texts:
        elem = elem.strip().strip('"').strip("'")
        elem = re.sub(r'\s+', ' ', elem)

        for punct in ["!", ".", "?", ':', ";", ","]:
            elem = elem.replace(f" {punct}", punct)
        elem = elem.replace("' ", "'")
        elem = elem.replace("#", '')

        elem = elem.replace("( ", "(")
        elem = elem.replace(" )", ")")

        ret.append(elem.lower())

    return ret


def main():
    now = datetime.now()

    parser = argparse.ArgumentParser()

    # Preferrably, set defaults for all arguments
    parser.add_argument('--texts', '-t', help='Path to texts for obfuscation',
                        default='./Data/testTuring_1.txt') # default set to a small sample data.
    parser.add_argument('--authors_total', '-at', help='Number of Total Authors in Corpus',
                        default=20) # default set to a small sample data.
    parser.add_argument('--dir', '-f', help='Path to the directory containing the trained model',
                        default='./Trained_Models/testTuring_06.09.20.33.32') # Default changes because each trial appends timestamp to the Train.py trial name
    parser.add_argument('--trial_name', '-tm', help='The Current Trial\'s Name (e.g. Dataset Name)', default='testTuring_Obfuscate')

    parser.add_argument('--L', '-L', help='L, the number of top POS n-grams to mask', default=15)
    parser.add_argument('--c', '-c', help='c, the length scaling constant', default=1.35)
    parser.add_argument('--min_length', '-min', help='The minimum length of POS n-gram to consider for obfuscation',
                        default=1.35)

    parser.add_argument('--ig_steps', '-ig', help='The number of steps for IG importance extraction', default=1024)


    args = parser.parse_args()

    dir = os.getcwd()
    timestamp = now.strftime("%m.%d.%H.%M.%S")
    save_path = os.path.join(dir, 'Trained_Models', f'{args.trial_name}_{timestamp}')

    os.makedirs(save_path)

    print('------------', '\n', 'Loading Data...')
    with open(args.texts, 'r') as reader:
        lines = [line.partition(' ') for line in reader.readlines()]
        data = pd.DataFrame(data = {
                                    'text' : [line[2] for line in lines],
                                    'label' : [int(line[0]) for line in lines]
                                    })

    data['POS'] = tag(data['text'])

    features = np.array(pickle.load(open(os.path.join(args.dir, 'features.pkl'), "rb")))
    Scaler= np.array(pickle.load(open(os.path.join(args.dir, 'Scaler.pkl'), "rb")))
    num_char = features[0].size
    num_pos = features[1].size

    # Flatten the features correctly
    features_flattened = [item for sublist in features for subsublist in sublist for item in subsublist]
    print(f"Expected input size for model: {len(features_flattened)}")

    print('------------', '\n', 'Data Iteration...')

    ngram_reps = []
    for idx, row in data.iterrows():
        ngram_reps.append(ngram_rep(row['text'], row['POS'], features))

    print('------------', '\n', 'Preprocessing...')
    Scaler = preprocessing.StandardScaler() # create an instance of StandardScaler and use its fit_transform method to scale the ngram_reps array
    ngram_reps = np.array(ngram_reps, dtype=object) # Add data type
    ngram_reps = Scaler.fit_transform(ngram_reps)

    ig_set = DataLoader(Loader(ngram_reps, data['label']), batch_size=1, shuffle=False)

    print('------------', '\n', 'Loading Models...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device selection in case Nvidia Cuda is not available
    model = Model(len(features_flattened), args.authors_total)
    model.load_state_dict(torch.load(os.path.join(args.dir, 'model.pt'), map_location=device), strict=False)
    model.to(device) # ensure model is moved to device
    model.eval()

    ig = IntegratedGradients(model)

    all_attributions = []
    torch.cuda.empty_cache()

    print('------------', '\n', 'Attribution Iteration...')
    for data_tensor, label_tensor in ig_set:
        data_tensor = data_tensor.to(device)
        label_tensor = label_tensor.to(torch.int64).to(device)

        attributions = ig.attribute(data_tensor, target=label_tensor, n_steps=args.ig_steps)
        attributions = attributions.tolist()

        for attribution in attributions:
            all_attributions.append(attribution)

        torch.cuda.empty_cache()
        del attributions

    # Create a new DataFrame for attributions and concatenate it with the original data DataFrame
    attributions_df = pd.DataFrame(all_attributions, columns=[f'attribution_{i}' for i in range(len(all_attributions[0]))])
    data = pd.concat([data.reset_index(drop=True), attributions_df.reset_index(drop=True)], axis=1)

    print('------------', '\n', 'isValid Function...')

    isValid = lambda index : index >= index >= num_char and index < num_char + num_pos
    to_compressed = lambda tag: tags[tag] if tag in tags else tag

    print('------------', '\n', 'Obfuscation in Progress...')
    obfuscated_texts = []
    for idx, row in data.iterrows():
        torch.cuda.empty_cache()
        text = row['text']

        # Retrieve attributions from the correct columns
        attribution = row[[f'attribution_{i}' for i in range(len(all_attributions[0]))]].values
        mult = [args.c ** len(feature) for feature in features_flattened]
        attribution = np.multiply(attribution, mult)

        ranked_indexes = np.argsort(np.array(attribution))
        ranked_indexes = [elem for elem in ranked_indexes if isValid(elem)]
        ranked_indexes.reverse()
        to_replace = [features_flattened[elem] for elem in ranked_indexes]

        to_replace = [replace for replace in to_replace if len(replace) > args.min_length]
        to_replace = to_replace[:args.L]

        words = tokenize(text)

        retagged = pos_tag(words)
        retagged = [to_compressed(tup[1]) for tup in retagged]

        intervals = []

        for replace in to_replace:

            starts = [i for i in range(len(retagged) - len(replace)) if replace == "".join(retagged[i:i + len(replace)])]

            for start in starts:
                intervals.append([start, start + len(replace)])

        changed = [False] * len(words)

        for interval in intervals:
            if not any(changed[interval[0] : interval[1]]):
                words = replace_interval(words, interval)
                changed[interval[0] : interval[1]] = [True] * (interval[1] - interval[0])

        obfuscated_texts.append(" ".join(words))

    print('------------', '\n', 'Writing Obfuscated Text...')

    obfuscated_texts = clean(obfuscated_texts)
    with open(os.path.join(save_path, 'adversarial_texts.txt'), 'w') as writer:
        writer.writelines(obfuscated_texts)

    print('------------', '\n', 'Obfuscated Text Written to document...')

if __name__ == "__main__":
    main()