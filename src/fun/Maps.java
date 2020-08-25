/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import javax.sound.midi.SysexMessage;

/**
 *
 * @author RAVI
 */
public class Maps {
    
    static class CustomClass{
        String key;

        public CustomClass(String key) {
            this.key = key;
        }

        public String getKey() {
            return key;
        }

        public void setKey(String key) {
            this.key = key;
        }
        
        
    }
    
    public static void main(String[] args) {
     
        Map<String, Integer> m = new HashMap<>();
        m.put("1_", 1);
        m.put("2_", 2);
        m.put("3_", 3);
        m.put(null, null);
        
//        Iterator keyIterator = m.keySet().iterator();
//        while(keyIterator.hasNext()){
//            System.out.println(" for keys: "+keyIterator.next());
//            System.out.println(" value is: "+m.get(keyIterator.next()));
//        }
        
//        Iterator<Map.Entry<String, Integer>> iterator = m.entrySet().iterator();
//        while(iterator.hasNext()){
//            System.out.println(iterator.next().getKey()+" : "+iterator.next().getValue());
//        }
        
        for (Map.Entry<String, Integer> e : m.entrySet()) {
            System.out.println(e.getKey() + " " + e.getValue());
        }
        
        m.put("3_", 67);
        
        for (Map.Entry<String, Integer> e : m.entrySet()) {
            System.out.println(e.getKey() + " " + e.getValue());
        }
        
        CustomClass a = new CustomClass("1_");
        CustomClass b = new CustomClass("2_");
        CustomClass c = new CustomClass("3_");
        
        System.out.println(a.hashCode());
        System.out.println(b.hashCode());
        System.out.println(c.hashCode());
        
        Map<CustomClass, Integer> n = new HashMap<>();
        n.put(a, 1);
        n.put(b, 2);
        n.put(c, 3);
        
        for (Map.Entry<CustomClass, Integer> e : n.entrySet()) {
            System.out.println(e.getKey() + " " + e.getValue());
        }
        
        n.put(c, 67);
        
        for (Map.Entry<CustomClass, Integer> e : n.entrySet()) {
            System.out.println(e.getKey() + " " + e.getValue());
        }
        
    }
}
