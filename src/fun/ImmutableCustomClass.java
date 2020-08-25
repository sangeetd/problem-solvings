/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

/**
 *
 * @author RAVI
 */

//make class final
final class OtherClass{
    
    //make all variable final
    private final String d;

    public OtherClass(String d) {
        this.d = d;
    }

    public String getD() {
        return d;
    }

    //do not expose setter methods
//    private void setD(String d) {
//        this.d = d;
//    }
    
    //when modifiable methods needs to be created
    //it should return new instance of the class
    public OtherClass append(String append){
        return new OtherClass(this.d+append);
    }
    
}

//inheritance not aallowed.
//class SubOtherClass extends OtherClass{
//    
//}

public class ImmutableCustomClass {
    
    public static void main(String[] args) {
        
        OtherClass c = new OtherClass("hello");
        System.out.println(c.getD());
        OtherClass newOc = c.append(" world");
        System.out.println(newOc.getD());
        
    }
    
}
