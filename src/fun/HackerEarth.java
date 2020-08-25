/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.lang.annotation.ElementType;
import java.lang.annotation.Repeatable;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author RAVI
 */
public class HackerEarth
{

 @Target({ElementType.TYPE_PARAMETER, ElementType.TYPE_USE})
 @interface MyAnnotation { }

 @Retention(RetentionPolicy.RUNTIME)
 @interface Hints 
 {
  Hint[] value();
 }

 @Repeatable(Hints.class)
 @Retention(RetentionPolicy.RUNTIME)
 @interface Hint
 {
  String value();
 }

 @Hint("hint1")
 @Hint("hint2")

 class Person { }

 public static void main(String[] args) 
  {
  Hint hint = Person.class.getAnnotation(Hint.class);
  System.out.println(hint);
  Hints hints1 = Person.class.getAnnotation(Hints.class);
  System.out.println(hints1.value().length);
  Hint[] hints2 = Person.class.getAnnotationsByType(Hint.class);
  System.out.println(hints2.length);
  
  //.....................................................
  int[] array = {6,9,8};
    List<Integer> list = new ArrayList<>();
    list.add(array[0]);
    list.add(array[2]);
    list.set(1, array[1]);
    list.remove(0);
    System.out.println(list);
//.....................................................
    
    ABB obj = new ABB();
    obj.cal(2, 3);
    
    System.out.println(obj.x + " " + obj.y);
  //.....................................................
    
  }
}
