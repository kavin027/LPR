import java.io.*;
import java.nio.file.*;
import java.util.*;
import meka.classifiers.multilabel.BR;
import weka.core.Instances;

/** Score only the original independent SMO arm; no classifier chains are fitted. */
public class IndependentSMO {
    public static void main(String[] args) throws Exception {
        Path work = Paths.get(args[0]);
        int nLabels = Integer.parseInt(args[1]);
        try (BufferedReader tr = Files.newBufferedReader(work.resolve("support.arff"));
             BufferedReader te = Files.newBufferedReader(work.resolve("query.arff"));
             BufferedReader options = Files.newBufferedReader(work.resolve("options.txt"));
             PrintWriter out = new PrintWriter(Files.newBufferedWriter(work.resolve("br.txt")))) {
            Instances train = new Instances(tr), test = new Instances(te);
            train.setClassIndex(nLabels); test.setClassIndex(nLabels);
            BR model = new BR();
            model.setClassifier(MatchedEnsemble.base());
            model.buildClassifier(new Instances(train));
            for (int i = 0; i < test.numInstances(); i++) {
                double[] prob = model.distributionForInstance(test.instance(i));
                int[] masks = Arrays.stream(options.readLine().split(" ")).mapToInt(Integer::parseInt).toArray();
                double[] ll = new double[masks.length];
                for (int j = 0; j < masks.length; j++)
                    for (int k = 0; k < nLabels; k++)
                        ll[j] += Math.log(((masks[j] >> k) & 1) == 1 ? prob[k] : 1-prob[k]);
                out.println(masks[MatchedEnsemble.argmax(ll)]);
            }
        }
    }
}
