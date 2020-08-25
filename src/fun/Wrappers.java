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
public class Wrappers {
    
    public static void main(String[] args) {
        
        int i = 1;
        
        //conversion of primitive to wrapper
        //boxing...............
        Integer iW = new Integer(i);
        
        //unboxing..................
        int j = iW.intValue();
        
        Integer iW_2 = new Integer(i);
        
        //primitive data comparision,value based
        if(i == j){
            System.out.println("true");
        }else{
            System.out.println("false");
        }
        
        //Object ref comparision which is false
        if(iW == iW_2){
            System.out.println("true");
        }else{
            System.out.println("false");
        }
        
        //value inside the object comparision,which are same in this case
        if(iW.equals(iW_2)){
            System.out.println("true");
        }else{
            System.out.println("false");
        }
        
        
        System.out.println("==/equals in case of autoboxing wrappers");
        
        //directly wrtting the value to wrapper
        //Auto boxing.............
        Double a = 0.2;
        Double b = 0.2;
        
        //Object ref comparision which is false
        if(a == b){
            System.out.println("true");
        }else{
            System.out.println("false");
        }
        
        //value inside the object comparision,which are same in this case
        if(a.equals(b)){
            System.out.println("true");
        }else{
            System.out.println("false");
        }
        
        Double c = b;
        
        if(c == b){
            System.out.println("true");
        }else{
            System.out.println("false");
        }
        
        //this is allowed
        int l = new Integer("1");
        //this is not allowed
        //int[] x = new Integer[]{1, 2, 3};
        //Integer[] x = new int[]{1, 2, 3};
        
    }
    
}
