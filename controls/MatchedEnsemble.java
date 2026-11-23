import java.io.*;
import java.nio.file.*;
import java.util.*;
import meka.classifiers.multilabel.MCC;
import meka.classifiers.multilabel.BR;
import weka.classifiers.functions.SMO;
import weka.core.*;

public class MatchedEnsemble extends MCC {
    static SMO base(){
        SMO m=new SMO();
        m.setC(1);
        m.setFilterType(new SelectedTag(SMO.FILTER_NONE,SMO.TAGS_FILTER));
        m.setBuildCalibrationModels(true);
        m.setRandomSeed(1);
        return m;
    }
    int[] order(){ return m_Chain; }
    double[] scoreAll(int[] candidates, Instance x) throws Exception {
        double[] out=new double[candidates.length];
        int[] prefixes=new int[candidates.length];
        int L=m_Chain.length;
        for(int depth=0;depth<L;depth++){
            int label=m_Chain[depth], bit=1<<label;
            Map<Integer,double[]> memo=new HashMap<>();
            for(int j=0;j<candidates.length;j++){
                int pf=prefixes[j];
                double[] c=memo.get(pf);
                if(c==null) {
                    double[] bits=new double[L];
                    for(int k=0;k<L;k++) bits[k]=(pf>>k)&1;
                    c=nodes[label].distribution(x,bits);
                    if(!Double.isFinite(c[0]+c[1]) || Math.abs(c[0]+c[1]-1)>1e-8)
                        throw new ArithmeticException("Invalid binary conditional");
                    memo.put(pf,c);
                }
                out[j]+=Math.log(c[(candidates[j]>>label)&1]);
                prefixes[j]|=candidates[j]&bit;
            }
        }
        return out;
    }
    static int argmax(double[] scores) {
        int b=0;
        for(int j=1;j<scores.length;j++) if(scores[j]>scores[b]) b=j;
        return b;
    }
    static void run(Path root,int L) throws Exception {
        Instances train,test;
        try(BufferedReader rd=Files.newBufferedReader(root.resolve("support.arff"))){train=new Instances(rd);}
        try(BufferedReader rd=Files.newBufferedReader(root.resolve("query.arff"))){test=new Instances(rd);}
        train.setClassIndex(L);test.setClassIndex(L);
        MatchedEnsemble[] ms=new MatchedEnsemble[10];
        BR br=new BR();br.setClassifier(base());br.buildClassifier(new Instances(train));
        try(PrintWriter orders=new PrintWriter(Files.newBufferedWriter(root.resolve("orders.txt")))){
            for(int seed=1;seed<=10;seed++){
                MatchedEnsemble m=new MatchedEnsemble();
                m.setSeed(seed);m.setChainIterations(0);m.setInferenceIterations(10);m.setClassifier(base());
                m.buildClassifier(new Instances(train));
                ms[seed-1]=m;
                orders.println(seed+" "+Arrays.toString(m.order()));
            }
        }
        try(BufferedReader options=Files.newBufferedReader(root.resolve("options.txt"));
            PrintWriter out=new PrintWriter(Files.newBufferedWriter(root.resolve("ensemble.csv")))){
            out.println("independent_smo,epcc,pcc_1,pcc_2,pcc_3,pcc_4,pcc_5,pcc_6,pcc_7,pcc_8,pcc_9,pcc_10");
            for(int i=0;i<test.numInstances();i++){
                int[] cand=Arrays.stream(options.readLine().split(" ")).mapToInt(Integer::parseInt).toArray();
                Instance x=test.instance(i);
                double[] probs=br.distributionForInstance(x), brScore=new double[cand.length];
                for(int j=0;j<cand.length;j++) for(int k=0;k<L;k++)
                    brScore[j]+=Math.log(((cand[j]>>k)&1)==1?probs[k]:1-probs[k]);
                double[][] logp=new double[10][];
                for(int t=0;t<10;t++) logp[t]=ms[t].scoreAll(cand,x);
                // Log-sum-exp minus an irrelevant log(10) yields a probability mixture.
                double[] mix=new double[cand.length];
                for(int j=0;j<cand.length;j++){
                    double m=Double.NEGATIVE_INFINITY;
                    for(double[] vec:logp) m=Math.max(m,vec[j]);
                    mix[j]=m;
                    if(Double.isFinite(m)){
                        double sum=0;for(double[] vec:logp)sum+=Math.exp(vec[j]-m);
                        mix[j]+=Math.log(sum);
                    }
                }
                out.print(cand[argmax(brScore)]+","+cand[argmax(mix)]);
                for(double[] vec:logp) out.print(","+cand[argmax(vec)]);
                out.println();
            }
            if(options.readLine()!=null)throw new AssertionError("Extra candidate records");
        }
    }
    public static void main(String[] args) {
        try { run(Paths.get(args[0]), Integer.parseInt(args[1])); }
        catch(Exception e) { throw new RuntimeException(e); }
    }
}
