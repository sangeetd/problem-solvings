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
public class Node<T> {
    
    private T data;
    private Node<T> previous;
    private Node<T> next;
    private Node<T> random;

    public Node(T data, Node previous, Node next) {
        this.data = data;
        this.previous = previous;
        this.next = next;
    }

    public Node(T data) {
        this.data = data;
        this.previous = null;
        this.next = null;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    public Node<T> getPrevious() {
        return previous;
    }

    public void setPrevious(Node<T> previous) {
        this.previous = previous;
    }

    public Node<T> getNext() {
        return next;
    }

    public void setNext(Node<T> next) {
        this.next = next;
    }

    public Node<T> getRandom() {
        return random;
    }

    public void setRandom(Node<T> random) {
        this.random = random;
    }
    
    
}
