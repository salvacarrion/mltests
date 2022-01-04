# fairseq-generate "" --source-lang de --target-lang es --path checkpoints/checkpoint_best.pt --scoring bleu --beam 5 --nbest 1 --max-len-a 1.2 --max-len-b 10
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)