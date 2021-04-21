/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 *
 * @author sangeetdas
 */
public class BinaryHeap<T> {

    private List<T> heapList;
    private Comparator<T> comparator;

    public BinaryHeap() {
        this.heapList = new ArrayList<>();
        this.comparator = (a, b) -> a.hashCode() - b.hashCode();
    }

    public BinaryHeap(Comparator<T> comparator) {
        this();
        this.comparator = comparator;
    }

    public void add(T data) {

        this.heapList.add(data);

        if (this.heapList.size() > 1) {
            heapify(this.heapList.size() - 1);
        }
    }

    private void heapify(int index) {

        int left = getLeftIndex(index);
        int right = getRightIndex(index);
        int tempIndex = index;
        System.out.println(left + " "+ right + " ");
        if(left < this.heapList.size() 
          && this.comparator.compare(this.heapList.get(left), this.heapList.get(index)) <= 0){
            tempIndex = left;
        }
        if(right < this.heapList.size() 
          && this.comparator.compare(this.heapList.get(right), this.heapList.get(tempIndex)) <= 0){
            tempIndex = right;
        }
        if(tempIndex != index){
            swap(index, tempIndex);
            heapify(tempIndex);
        }

    }

    public void print(){
        System.out.println(this.heapList);
    }
    
    private void swap(int i, int j) {
        T temp = this.heapList.get(i);
        this.heapList.set(i, this.heapList.get(j));
        this.heapList.set(j, temp);
    }

    private int getParentIndex(int index) {
        return (index - 1) / 2;
    }

    private int getLeftIndex(int index) {
        return (2 * index) + 1;
    }

    private int getRightIndex(int index) {
        return (2 * index) + 2;
    }
}
