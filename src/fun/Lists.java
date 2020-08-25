/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 *
 * @author RAVI
 */

class CustomClassTest implements Comparable<CustomClassTest>{
    
    int id;
    String str;

    public CustomClassTest(int id, String str) {
        this.id = id;
        this.str = str;
    }

    public int getId() {
        return id;
    }

    public String getStr() {
        return str;
    }

    @Override
    public int compareTo(CustomClassTest o) {
        
        System.out.println("invoking Comparable compareTo() by sort()");
        
        return (this.getId() < o.getId() ? -1 : (this.getId() == o.getId() ? 0 : 1));
        
    }
    
}

class CustomClassTest2{
    
    int id;
    String str;

    public CustomClassTest2(int id, String str) {
        this.id = id;
        this.str = str;
    }

    public int getId() {
        return id;
    }

    public String getStr() {
        return str;
    }
    
    public static final Comparator<CustomClassTest2> idComparator = new Comparator<CustomClassTest2>(){

        @Override
        public int compare(CustomClassTest2 o1, CustomClassTest2 o2) {
            
            System.out.println("invoking id Comparator compare() by sort()");
        
            return (o1.getId() < o2.getId() ? -1 : (o1.getId() == o2.getId() ? 0 : 1));
            
        }
        
    };
    
    public static final Comparator<CustomClassTest2> strComparator = new Comparator<CustomClassTest2>(){

        @Override
        public int compare(CustomClassTest2 o1, CustomClassTest2 o2) {
            
            System.out.println("invoking str Comparator compare() by sort()");
        
            return (int) (o1.getStr().compareTo(o1.getStr()));
            
        }
        
    };    
    
}

public class Lists {
    
    
    public static void main(String[] args) {
        
        List<Integer> i = new ArrayList<>(
                Arrays.asList(1, 2, 3, 4, 5
                )
        
        );
        
        i.stream().forEach( a -> System.out.println(a));
        
        i.forEach( a -> System.out.println(a));
        
        System.out.println("Empty list j");
        
        List<Integer> j = new ArrayList<>();
        
        j.stream().forEach( a -> System.out.println(a));
        
        j.forEach( a -> System.out.println(a));
        
        System.out.println("null list k");
        
        List<Integer>k = null;
        try{
            System.out.println("null list k-stream-foreach");
            k.stream().forEach( a -> System.out.println(a));
        }catch(Exception e){
            e.printStackTrace();
        }
        
        try{
            System.out.println("null list k-list-foreach");
            k.forEach( a -> System.out.println(a));
        }catch(Exception e){
            e.printStackTrace();
        }
        
        System.out.println("Collections sorting");
        
        List<Integer> intList = new ArrayList<>(
                Arrays.asList(32, 64, 50, 1, -1, 29, 2, 111
                )
        
        );
        
        Collections.sort(intList);
        System.out.println("asc sort order: "+ intList);
        
        Collections.sort(intList, Collections.reverseOrder());
        System.out.println("desc sort order: "+ intList);
        
        System.out.println("List with custom objects for Comparable interface");
        
        List<CustomClassTest> objList = new ArrayList<>(
                Arrays.asList(
                        new CustomClassTest(1, "hello"),
                        new CustomClassTest(2, "java"),
                        new CustomClassTest(3, "c"),
                        new CustomClassTest(4, "python"),
                        new CustomClassTest(5, "c#")
                )
        
        );
        
        System.out.println("printing list of objects");
        System.out.println(objList);
        
        System.out.println("sorting list of objects without implementing comparator");
        //Collections.sort(objList); //errorneous
        
        System.out.println("sorting list of objects implementing comparator");
        //implement Comparable interface to your custom class
        //sort overloaded method in Collections class accepts those objects 
        //that has implemneted Comparable 
        Collections.sort(objList);
        System.out.println(objList);
        
        
        System.out.println("List with custom objects for Comparator interface");
        
        List<CustomClassTest2> objList2 = new ArrayList<>(
                Arrays.asList(
                        new CustomClassTest2(1, "hello"),
                        new CustomClassTest2(2, "java"),
                        new CustomClassTest2(3, "c"),
                        new CustomClassTest2(4, "python"),
                        new CustomClassTest2(5, "c#")
                )
        
        );
        
        System.out.println("printing list of objects");
        System.out.println(objList2);
        
        System.out.println("sorting list of objects without implementing comparator");
        //Collections.sort(objList2); //errorneous
        
        System.out.println("sorting list of objects implementing comparator");
        //implement Comaparator interface as inner anonymous class to your custom class
        //sort overloaded method in Collections class accepts Comparator implementing object ref as second parameter
        Collections.sort(objList2, CustomClassTest2.idComparator);
        System.out.println(objList2);
        
        Collections.sort(objList2, CustomClassTest2.strComparator);
        System.out.println(objList2);
        
    }
    
}
