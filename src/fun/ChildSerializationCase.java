/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 *
 * @author RAVI
 */
class Parent {

    int i;

    public Parent(){
        //no-args default constructor
        
        //if you don't provide default value in no-args default constructor when jvm call it
        //it will auto assign it with variable's default value (0/null: primitive type /Object ref variable type)
        this.i=1000; //some default value otherwise 0 will be assigned
        System.out.println("no-args default constructor will be called by Jvm at de-serilization time...");
    }
    
    public Parent(int i) {
        this.i = i;
    }

}

class Child extends Parent implements Serializable {

    int j;

    public Child(int i, int j) {
        super(i);
        this.j = j;
    }

}

public class ChildSerializationCase {

    /*
    
    if child implements Serializable and parent doesn't then these 2 things happen
    
    Serialization: At the time of serialization, if any instance variable is inheriting from non-serializable superclass(parent), 
    then JVM ignores original value of that instance variable and save default value to the file.
    
    De- Serialization: At the time of de-serialization, if any non-serializable superclass is present, 
    then JVM will execute instance control flow in the superclass. To execute instance control flow in a class, 
    JVM will always invoke default(no-arg) constructor of that class. 
    So every non-serializable superclass must necessarily contain default constructor, 
    otherwise we will get runtime-exception.
    
    if parent doesn't provide the no args default constructor this exception occurs
    
    Exception in thread "main" java.io.InvalidClassException: fun.Child; no valid constructor
	at java.io.ObjectStreamClass$ExceptionInfo.newInvalidClassException(ObjectStreamClass.java:157)
	at java.io.ObjectStreamClass.checkDeserialize(ObjectStreamClass.java:862)
	at java.io.ObjectInputStream.readOrdinaryObject(ObjectInputStream.java:2041)
	at java.io.ObjectInputStream.readObject0(ObjectInputStream.java:1571)
	at java.io.ObjectInputStream.readObject(ObjectInputStream.java:431)
	at fun.ChildSerializationCase.main(ChildSerializationCase.java:68)
Java Result: 1
    
    
    
    */
    
    
    
    public static void main(String[] args)
            throws Exception {
        Child c1 = new Child(10, 20);

        System.out.println("i = " + c1.i);
        System.out.println("j = " + c1.j);

        // Serializing B's(subclass) object  
        //Saving of object in a file 
        FileOutputStream fos = new FileOutputStream("child-serialization-case.txt");
        ObjectOutputStream oos = new ObjectOutputStream(fos);

        // Method for serialization of B's class object 
        oos.writeObject(c1);

        // closing streams 
        oos.close();
        fos.close();

        System.out.println("Object has been serialized");

        // De-Serializing B's(subclass) object  
        // Reading the object from a file 
        FileInputStream fis = new FileInputStream("child-serialization-case.txt");
        ObjectInputStream ois = new ObjectInputStream(fis);

        // Method for de-serialization of B's class object 
        Child c2 = (Child) ois.readObject();

        // closing streams 
        ois.close();
        fis.close();

        System.out.println("Object has been deserialized");

        System.out.println("i = " + c2.i);
        System.out.println("j = " + c2.j);
    }

}
