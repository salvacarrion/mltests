import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from autonmt.bundle import utils
import pandas as pd
sns.set()

from autonmt.preprocessing import DatasetBuilder

import numpy as np
from sklearn.manifold import TSNE

from autonmt.bundle import utils

main_words = ["<unk>","<s>","</s>","<pad>","a",".","in","the","on","man","is","and","of","with","woman",",","two","are","to","people","at","an","wearing","young","white","shirt","black","his","while","blue","men","sitting","girl","red","boy","dog","standing","playing","group","street","down","walking","front","her","holding","one","water","three","by","women","green","little","up","for","child","looking","outside","as","large","through","yellow","children","brown","person","from","their","ball","hat","into","small","next","other","dressed","some","out","over","building","riding","running","near","jacket","another","around","sidewalk","field","orange","crowd","beach","stands","pink","sits","jumping","behind","table","grass","background","snow","bike","stand","city","&apos;s","girls","air","player","asian","looks","wall","top","dogs","several","that","older","four","off","dress","camera","park","talking","lady","something","soccer","along","walks","guitar","boys","hair","play","together","working","food","smiling","gray","picture","has","game","plays","car","holds","hand","it","road","him","bench","glasses","pants","old","shorts","stage","sit","carrying","walk","baby","couple","them","bicycle","side","face","&quot;","male","tree","pool","race","taking","rock","each","doing","across","watching","guy","dirt","head","jeans","there","area","blond","jumps","boat","hands","female","day","ground","performing","room","back","baseball","who","eating","being","football","coat","using","kids","suit","under","band","striped","watch","many","horse","mouth","purple","he","sign","store","long","this","runs","sand","tennis","players","construction","look","sunglasses","reading","clothing","microphone","mountain","basketball","dark","ocean","toy","during","its","workers","middle","watches","t-shirt","uniform","climbing","past","elderly","restaurant","against","helmet","team","train","dancing","window","rides","they","about","chair","work","be","posing","trees","having","outdoor","wooden","covered","five","waiting","swimming","all","getting","or","floor","trying","very","colorful","skateboard","bag","busy","ice","fence","singing","shirts","jump","laying","line","hill","market","cart","book","bright","hats","inside","ride","tan","truck","cap","kitchen","others","cellphone","path","big","grassy","high","someone","making","bus","clothes","motorcycle","takes","umbrella","outfit","towards","enjoying","full","night","track","body","brick","light","metal","river","swing","paper","ready","she","tank","shop","open","piece","sweater","trick","above","worker","lake","adults","going","painting","colored","music","dance","hard","snowy","run","shopping","surrounded","wave","onto","uniforms","vest","outdoors","stick","photo","stone","backpack","beside","crowded","smiles","board","kid","african","drinking","gear","subway","family","hockey","phone","pole","american","house","toddler","event","gathered","guys","hanging","police","arms","bridge","catch","flowers","set","beautiful","preparing","away","fire","costume","leaning","object","sleeping","after","steps","fishing","lot","machine","fountain","plaid","rope","bar","shirtless","adult","take","like","stairs","rocks","shoes","arm","graffiti","selling","sunny","does","forest","parade","racing","setting","chairs","pose","tall","both","putting","volleyball","corner","frisbee","glass","pushing","throwing","between","drink","have","ladies","slide","computer","laughing","public","short","equipment","flag","instruments","couch","pictures","cowboy","party","plastic","ramp","wood","playground","poses","sweatshirt","trail","view","which","woods","beard","concrete","fish","statue","sun","dock","midair","winter","yard","number","bikes","seated","skirt","attire","just","reads","toward","beer","cooking","distance","get","what","works","blond-hair","cutting","rider","six","bags","buildings","left","pulling","school","vests","where","cream","middle-aged","appears","bed","cliff","skateboarder","sky","station","cross","crossing","edge","few","friends","instrument","mountains","showing","spectators","cars","horses","jersey","mother","tent","bearded","court","filled","make","nearby","performs","right","scarf","apron"]
main_words = set(main_words[:int(len(main_words)/2)])


def main():
    file = "trg"
    path = "/home/scarrion/Documents/Programming/Python/mltests/translation/autonmt/.outputs/tmp/256/multi30k_de-en_original_word_8000/"

    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            # {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[250, 500, 1000, 2000, 4000, 8000],
        merge_vocabs=False,
        force_overwrite=False,
        use_cmd=True,
        eval_mode="same",
        conda_env_name="mltests",
        letter_case="lower",
    ).build(make_plots=False, safe=True)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    train_tsne = False
    file = "trg"
    for origin_emb_size in [256, 512]:
        for ds in tr_datasets:
            base_path = f".outputs/tmp/{origin_emb_size}/{str(ds)}"

            if train_tsne:
                x = np.load(os.path.join(base_path, f"{file}.npy"))
                x_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x)
                np.save(os.path.join(base_path, f"{file}_tsne.npy"), x_embedded)
                print(f"File saved! ({str(ds)})")
            else:
                x = np.load(os.path.join(base_path, f"{file}_tsne.npy"))
                labels = utils.read_file_lines(ds.get_vocab_file("en") + ".vocab")
                labels = [l.split('\t')[0] for l in labels]
                data = pd.DataFrame(data=x, columns=["f1", "f2"])
                data["label"] = labels

                scale = 2.0
                plt.figure(figsize=(12, 12))
                sns.set(font_scale=scale)

                g = sns.scatterplot(
                    x="f1", y="f2",
                    palette=sns.color_palette("hls", 10),
                    data=data,
                    legend="full",
                    alpha=0.3
                )
                # g.set(title=str(ds).replace('_', ' ') + f"\n(source emb. {origin_emb_size})")

                for i, row in data.iterrows():
                    word = row["label"].replace('▁', '')
                    if word in main_words:
                        g.annotate(word, (row["f1"], row["f2"]), fontsize=8*scale)
                plt.tight_layout()

                # Print plot
                savepath = os.path.join(base_path, "plots")
                utils.make_dir(savepath)
                for ext in ["png", "pdf"]:
                    path = os.path.join(savepath, f"tsne_{file}__{str(ds)}.{ext}")
                    plt.savefig(path, dpi=300)
                    print(f"Plot saved! ({path})")

                # plt.show()
                plt.close()
                asdas = 3


if __name__ == "__main__":
    main()
