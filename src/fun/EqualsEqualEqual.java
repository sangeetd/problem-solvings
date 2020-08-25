/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author RAVI
 */
public class EqualsEqualEqual {
    
    public static void main(String[] args) {
        
        String s1="hello";
        String s2 = "hello";
        
        System.out.println("s1 == s2 "+(s1==s2));
        System.out.println("s1 equals s2 "+(s1.equals(s2)));
        
        String s3 = new String("hello");
        
        System.out.println("s2 == s3 "+(s2==s3));
        System.out.println("s2 equals s3 "+(s2.equals(s3)));
        
        System.out.println("String reverse");
        String str = "Java Is Love";
        
        for(int i=str.length()-1; i>=0; i--){
            System.out.print(str.charAt(i)+" ");
        }
        
        
        
    }
    
}
