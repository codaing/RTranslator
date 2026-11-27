package nie.translator.seamless;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Kaldi-style 80-bin log mel frontend. Exact fairseq2 parity remains a CI gate. */
public final class StreamingFbank {
    public static final int RATE=16000, WINDOW=400, HOP=160, BINS=80, FFT=512;
    private short[] residual=new short[0];
    private final double[][] weights=new double[BINS][FFT/2+1];
    private final double[] window=new double[WINDOW];
    public StreamingFbank() {
        double lo=mel(20), hi=mel(8000);
        for(int m=0;m<BINS;m++) {
            double a=lo+(hi-lo)*m/(BINS+1), b=lo+(hi-lo)*(m+1)/(BINS+1), c=lo+(hi-lo)*(m+2)/(BINS+1);
            for(int k=0;k<=FFT/2;k++){
                double x=mel(k*RATE/(double)FFT);
                weights[m][k]=Math.max(0,Math.min((x-a)/(b-a),(c-x)/(c-b)));
            }
        }
        for(int i=0;i<WINDOW;i++) window[i]=Math.pow(0.5-0.5*Math.cos(2*Math.PI*i/(WINDOW-1)),0.85);
    }
    private static double mel(double hz){return 1127*Math.log1p(hz/700);}
    public List<float[]> push(short[] pcm) {
        short[] data=Arrays.copyOf(residual,residual.length+pcm.length);
        System.arraycopy(pcm,0,data,residual.length,pcm.length);
        List<float[]> out=new ArrayList<>();
        int offset=0;
        while(offset+WINDOW<=data.length){out.add(frame(data,offset));offset+=HOP;}
        residual=Arrays.copyOfRange(data,offset,data.length);
        return out;
    }
    private float[] frame(short[] data,int offset){
        double[] re=new double[FFT], im=new double[FFT];
        double mean=0;
        for(int i=0;i<WINDOW;i++)mean+=data[offset+i];
        mean/=WINDOW;
        for(int i=0;i<WINDOW;i++){
            double cur=data[offset+i]-mean;
            double prev=i==0?cur:data[offset+i-1]-mean;
            re[i]=(cur-0.97*prev)*window[i];
        }
        for(int i=1,j=0;i<FFT;i++){
            int bit=FFT>>1;
            for(;(j&bit)!=0;bit>>=1)j^=bit;
            j^=bit;
            if(i<j){double x=re[i];re[i]=re[j];re[j]=x;}
        }
        for(int len=2;len<=FFT;len<<=1){
            double theta=-2*Math.PI/len;
            for(int i=0;i<FFT;i+=len)for(int j=0;j<len/2;j++){
                int a=i+j,b=a+len/2;
                double cos=Math.cos(theta*j),sin=Math.sin(theta*j);
                double x=re[b]*cos-im[b]*sin,y=re[b]*sin+im[b]*cos;
                re[b]=re[a]-x;im[b]=im[a]-y;re[a]+=x;im[a]+=y;
            }
        }
        float[] result=new float[BINS];
        for(int m=0;m<BINS;m++){
            double e=0;
            for(int k=0;k<=FFT/2;k++)e+=weights[m][k]*(re[k]*re[k]+im[k]*im[k]);
            result[m]=(float)Math.log(Math.max(e,Math.ulp(1.0f)));
        }
        return result;
    }
}
