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
public class LinkedListUtil<T> {

    private Node head;
    private Node tail;
    private long size = 0;

    public LinkedListUtil() {
    }
    
    public LinkedListUtil(Node<T> node) {
        head = node;
        
        //traverse to end and set tail
        //singly linked list
        Node<T> temp = head;
        while(temp.getNext() != null){
            size++;
            temp = temp.getNext();
            
        }
        
        tail = temp;
        size++;
        
    }

    public void addAtHead(T data) {

        if (head == null && tail == null) {
            Node<T> n = new Node<>(data);
            head = tail = n;
            size++;
            return;
        }

        Node<T> n = new Node<>(data);
        n.setNext(head);
        head.setPrevious(n);
        head = n;
        size++;

    }

    public void addAtTail(T data) {

        if (head == null && tail == null) {
            Node<T> n = new Node<>(data);
            head = tail = n;
            size++;
            return;
        }

        Node<T> n = new Node<>(data);
        n.setPrevious(tail);
        tail.setNext(n);
        tail = n;
        size++;

    }

    public void add(T data, int index) {

        if (head == null && tail == null) {
            Node<T> n = new Node<>(data);
            head = tail = n;
            size++;
            return;
        }

        if (index > length()) {
            addAtTail(data);
            return;
        }

        int nodeIndexer = 1;
        Node<T> n = new Node<>(data);
        Node<T> temp = head;

        while (temp != null) {

            if (nodeIndexer == index) {

                Node<T> prev = temp.getPrevious();

                prev.setNext(n);
                n.setPrevious(prev);
                n.setNext(temp);
                temp.setPrevious(n);
                size++;
                break;
            }

            nodeIndexer++;
            temp = temp.getNext();

        }

    }

    public void append(T data) {

        if (head == null && tail == null) {
            Node<T> n = new Node<>(data);
            head = tail = n;
            size++;
            return;
        }

        Node<T> n = new Node<>(data);
        n.setPrevious(tail);
        tail.setNext(n);
        tail = n;
        size++;
    }

    public void deleteAtHead() {

        if (head == null && tail == null) {
            throw new RuntimeException("LinkedList is already empty");
        }

        if (length() == 1) {
            clear();
            return;
        }

        Node<T> headNext = head.getNext();
        headNext.setPrevious(null);
        head.setNext(null);
        head = headNext;
        size--;

    }

    public void deleteAtTail() {

        if (head == null && tail == null) {
            throw new RuntimeException("LinkedList is already empty");
        }

        if (length() == 1) {
            clear();
            return;
        }

        Node<T> tailPrevious = tail.getPrevious();
        tailPrevious.setNext(null);
        tail.setPrevious(null);
        tail = tailPrevious;
        size--;

    }

    public void delete(int index) {

        if (head == null && tail == null) {
            throw new RuntimeException("LinkedList is already empty");
        }

        if (length() == 1) {
            clear();
            return;
        }

        if (index == length()) {
            deleteAtTail();
            return;
        }

        if (index == 1) {
            deleteAtHead();
            return;
        }

        int nodeIndexer = 1;
        Node<T> temp = head;

        while (temp != null) {

            if (nodeIndexer == index) {

                Node<T> prev = temp.getPrevious();
                Node<T> next = temp.getNext();

                prev.setNext(next);
                next.setPrevious(prev);

                temp.setPrevious(null);
                temp.setNext(null);

                size--;

                break;

            }

            nodeIndexer++;
            temp = temp.getNext();

        }

    }

    public void delete() {

        if (head == null && tail == null) {
            throw new RuntimeException("LinkedList is already empty");
        }

        if (length() == 1) {
            clear();
            return;
        }

        Node<T> headNext = head.getNext();
        headNext.setPrevious(null);
        head.setNext(null);
        head = headNext;
        size--;

    }

    public T get(int index) {

        if (head == null && tail == null) {
            throw new RuntimeException("LinkedList is already empty");
        }

        if (length() == 1) {
            return (T) head.getData();
        }

        int nodeIndexer = 1;
        Node<T> temp = head;

        while (temp != null) {

            if (nodeIndexer == index) {
                return (T) temp.getData();
            }

            nodeIndexer++;
            temp = temp.getNext();

        }
        
        return null;

    }

    public Node<T> search(T data) {

        if (head == null && tail == null) {
            throw new RuntimeException("LinkedList is already empty");
        }

        Node<T> temp = head;
        while (temp != null) {

            if (temp.getData().equals(data)) {
                return temp;
            }

            temp = temp.getNext();

        }

        return temp;
    }

    public void print() {

        if (head == null && tail == null) {
            throw new RuntimeException("LinkedList is already empty");
        }

        Node<T> temp = head;
        while (temp != null) {

            System.out.print(temp.getData());

            temp = temp.getNext();

        }

        System.out.println();

    }

    public long length() {
        return size;
    }

    public LinkedListUtil<T> reverse() {

        LinkedListUtil<T> newLL = new LinkedListUtil<>();

        Node<T> temp = tail;

        while (temp != null) {

            newLL.append(temp.getData());

            temp = temp.getPrevious();

        }

        return newLL;
    }

    public void clear() {
        head = tail = null;
        size = 0;
    }
    
    
}
