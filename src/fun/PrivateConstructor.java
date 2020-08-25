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
class A{

    //comment and A will not be inheritable
    public A(int a){
        //consturctor chaining
        //also this() must be the first line in this case
        this();
        System.out.println(a);
        
    }
    //comment and A will not be inheritable
    
    private A() {
        System.out.println("Private");
    }
    
}

class B extends A{

    public B() {
        super(2);
    }
    
    
}

public class PrivateConstructor {
    
    public static void main(String[] args) {
        B b=new B();
    }
    
}


