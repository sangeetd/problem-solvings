/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.Stack;

/**
 *
 * @author RAVI
 */
public class QueueImplmentationUsingTwoStacks {
    
    private final Stack<Integer> s1;
    private final Stack<Integer> s2;

    public QueueImplmentationUsingTwoStacks() {
        this.s1 = new Stack<>();
        this.s2 = new Stack<>();
    }

    public void enQueue(int x){
        
        if(s1.isEmpty()){
            s1.push(x);
        }else{
            //empty s1 to s2
            while(!s1.isEmpty()){
                s2.push(s1.pop());
            }
            //push x to s1
            s1.push(x);
            //push back s2 to s1
            while(!s2.isEmpty()){
                s1.push(s2.pop());
            }
        }
        
    }
    
    public int deQueue(){
        
        if(s1.isEmpty()){
            return -1;
        }else {
            return s1.pop();
        }
        
    }
    
    public int peek(){
        if(s1.isEmpty()){
            return -1;
        }else {
            return s1.peek();
        }
    }
    
    public void print(){
        System.out.println("print: Rear-> " +s1+" <- Front");
    }
    
    
}
