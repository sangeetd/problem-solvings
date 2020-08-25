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
public class QuickSort {

    public static int getPivot(int[] a, int f, int l){
        
        int pivot = l;
        int i=f-1;
        
        for(int j=f; j<=l; j++){
            
            if(a[j]<=a[pivot]){
                i++;
                int temp = a[j];
                a[j]=a[i];
                a[i]=temp;
            }
            
        }
        
        return i;
        
    }
    
    public static void quickSort(int[] a, int f, int l) {

        if (f <= l) {

            int pivot = getPivot(a, f, l);
            
            quickSort(a, f, pivot-1);
            quickSort(a, pivot+1, l);
            
            
        }

    }

    public static void main(String[] args) {

        int[] a = {9, 4, 3, 6, 7, 1, 2, 11, 5};
        
        quickSort(a, 0, a.length-1);
        
        for(int x: a){
            System.out.print(x+" ");
        }
        
    }

}
