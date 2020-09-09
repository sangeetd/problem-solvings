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
public class ABB {
    
    
    final int var;
    
    static final int statVar = 1000;

    public ABB() {
        this.var = 10;
    }
    
    public ABB(int var) {
        this.var = var;
    }
    
    public int x;

    protected int y;

    void cal(int a, int b){

        x =  a + 1;

        y =  b;

    }     
    
    public static void main(String[] args) {
        
        ABB ob1 = new ABB();
        System.out.println(ob1.var);
        ABB ob2 = new ABB();
        System.out.println(ob2.var);
        
        ABB ob3 = new ABB(20);
        System.out.println(ob3.var);
        ABB ob4 = new ABB(50);
        System.out.println(ob4.var);
        
    }

    
}
