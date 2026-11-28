package nie.translator.seamless;
import org.junit.Test;
import static org.junit.Assert.*;
public class BenchmarkScheduleTest {
    @Test public void baselineMatchesOldConditions(){
        for(int count=0;count<=3000;count+=2)for(int prev=0;prev<=count;prev+=64){
            assertEquals(count>=2&&count>=192&&count-prev>=64,StreamingPolicy.shouldUpdate(count,prev,64,false));
            assertEquals(count>=2,StreamingPolicy.shouldUpdate(count,prev,64,true));
        }
    }
    @Test public void fastAndEofSchedules(){
        assertFalse(StreamingPolicy.shouldUpdate(192,0,256,false));
        assertTrue(StreamingPolicy.shouldUpdate(256,0,256,false));
        assertFalse(StreamingPolicy.shouldUpdate(510,256,256,false));
        assertTrue(StreamingPolicy.shouldUpdate(512,256,256,false));
        assertFalse(StreamingPolicy.shouldUpdate(1334,0,0,false));
        assertTrue(StreamingPolicy.shouldUpdate(1334,0,0,true));
        assertFalse(StreamingPolicy.shouldUpdate(0,0,0,true));
    }
}
