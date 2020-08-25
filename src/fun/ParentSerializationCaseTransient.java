/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.NotSerializableException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 *
 * @author RAVI
 */

class Parent2 implements Serializable {

    int i;

    public Parent2(int i) {
        this.i = i;
    }

}

class Child2 extends Parent2 {

    transient int j;
   
    public Child2(int i, int j) {
        super(i);
        this.j = j;
    }

}

public class ParentSerializationCaseTransient {
    
    public static void main(String[] args)
            throws Exception {
        
        //parent that has implemented serializable is able to be serialized separately
        //but child is not
        
        Parent2 p1 = new Parent2(10);

        System.out.println("i = " + p1.i);

        // Serializing Child1's(subclass) object  
        //Saving of object in a file 
        FileOutputStream fos_ = new FileOutputStream("parent-serialization-case-transient-parentObject.txt");
        ObjectOutputStream oos_ = new ObjectOutputStream(fos_);

        // Method for serialization of B's class object 
        oos_.writeObject(p1);

        // closing streams 
        oos_.close();
        fos_.close();
        
        System.out.println("Parent Object has been serialized");
        
        
        // De-Serializing B's(subclass) object  
        // Reading the object from a file 
        FileInputStream fis_ = new FileInputStream("parent-serialization-case-transient-parentObject.txt");
        ObjectInputStream ois_ = new ObjectInputStream(fis_);

        // Method for de-serialization of B's class object 
        Parent2 p2 = (Parent2) ois_.readObject();

        // closing streams 
        ois_.close();
        fis_.close();

        System.out.println("Parent Object has been deserialized");

        System.out.println("i = " + p2.i);
        
        
        System.out.println("Attemping for child objects");
        
        Child2 c1 = new Child2(10, 20);

        System.out.println("i = " + c1.i);
        System.out.println("j = " + c1.j);

        // Serializing Child1's(subclass) object  
        //Saving of object in a file 
        FileOutputStream fos = new FileOutputStream("parent-serialization-case-transient-childObject.txt");
        ObjectOutputStream oos = new ObjectOutputStream(fos);

        // Method for serialization of B's class object 
        oos.writeObject(c1);

        // closing streams 
        oos.close();
        fos.close();

        System.out.println("Child Object has been serialized");
        
        // De-Serializing B's(subclass) object  
        // Reading the object from a file 
        FileInputStream fis = new FileInputStream("parent-serialization-case-transient-childObject.txt");
        ObjectInputStream ois = new ObjectInputStream(fis);

        // Method for de-serialization of B's class object 
        Child2 c2 = (Child2) ois.readObject();

        // closing streams 
        ois.close();
        fis.close();

        System.out.println("Child Object has been deserialized");

        System.out.println("i = " + c2.i);
        System.out.println("j = " + c2.j);
    }
    
    
}
