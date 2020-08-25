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
class Parent1 implements Serializable {

    int i;

    public Parent1(int i) {
        this.i = i;
    }

}

class Child1 extends Parent1 {

    int j;
   
    public Child1(int i, int j) {
        super(i);
        this.j = j;
    }

    //https://www.geeksforgeeks.org/object-serialization-inheritance-java/
    //https://www.journaldev.com/2452/serialization-in-java
    // By implementing writeObject method,  
    // we can prevent 
    // subclass from serialization 
    private void writeObject(ObjectOutputStream out) throws IOException {
        //below sysout proves that jvm invoke these writeObject and readObject
        //specific to serialization at runtime 
        //that why if we give our implemention of these method them jvm will run this one
        System.out.println("ObjectOutputStream that writes over Childs object is same called by JVM invocation: "+out);
        throw new NotSerializableException();
    }

    // By implementing readObject method,  
    // we can prevent 
    // subclass from de-serialization 
    private void readObject(ObjectInputStream in) throws IOException {
        throw new NotSerializableException();
    }

}

public class ParentSerializationCase {

    /*
    
     If the superclass is serializable but we don’t want the subclass to be serialized : 
    
     There is no direct way to prevent subclass from serialization in java. 
     One possible way by which a programmer can achieve this is by implementing the writeObject() and readObject() 
     methods in the subclass and needs to throw NotSerializableException from these methods. 
     These methods are executed during serialization and de-serialization respectively. 
     By overriding these methods, we are just implementing our own custom serialization.
    
     */
    public static void main(String[] args)
            throws Exception {
        
        //parent that has implemented serializable is able to be serialized separately
        //but child is not
        
        Parent1 p1 = new Parent1(10);

        System.out.println("i = " + p1.i);

        // Serializing Child1's(subclass) object  
        //Saving of object in a file 
        FileOutputStream fos_ = new FileOutputStream("parent-serialization-case-parentObject.txt");
        ObjectOutputStream oos_ = new ObjectOutputStream(fos_);

        // Method for serialization of B's class object 
        oos_.writeObject(p1);

        // closing streams 
        oos_.close();
        fos_.close();
        
        System.out.println("Parent Object has been serialized");
        
        
        // De-Serializing B's(subclass) object  
        // Reading the object from a file 
        FileInputStream fis_ = new FileInputStream("parent-serialization-case-parentObject.txt");
        ObjectInputStream ois_ = new ObjectInputStream(fis_);

        // Method for de-serialization of B's class object 
        Parent1 p2 = (Parent1) ois_.readObject();

        // closing streams 
        ois_.close();
        fis_.close();

        System.out.println("Parent Object has been deserialized");

        System.out.println("i = " + p2.i);
        
        
        System.out.println("Attemping for child objects");
        
        Child1 c1 = new Child1(10, 20);

        System.out.println("i = " + c1.i);
        System.out.println("j = " + c1.j);

        // Serializing Child1's(subclass) object  
        //Saving of object in a file 
        FileOutputStream fos = new FileOutputStream("parent-serialization-case-childObject.txt");
        ObjectOutputStream oos = new ObjectOutputStream(fos);

        // Method for serialization of B's class object 
        System.out.println("ObjectOutputStream that writes over Childs object: "+oos);
        oos.writeObject(c1);

        // closing streams 
        oos.close();
        fos.close();

        System.out.println("Child Object has been serialized");
        
        // De-Serializing B's(subclass) object  
        // Reading the object from a file 
        FileInputStream fis = new FileInputStream("parent-serialization-case-childObject.txt");
        ObjectInputStream ois = new ObjectInputStream(fis);

        // Method for de-serialization of B's class object 
        Child1 c2 = (Child1) ois.readObject();

        // closing streams 
        ois.close();
        fis.close();

        System.out.println("Child Object has been deserialized");

        System.out.println("i = " + c2.i);
        System.out.println("j = " + c2.j);
    }

}
