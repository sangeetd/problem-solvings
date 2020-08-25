/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.Stack;

/**
 *
 * @author RAVI
 */
public class SpecialStack {

    private Stack<Integer> s;
    private int minE;

    public SpecialStack() {
        this.s = new Stack<>();
    }

    public void push(int x) {

        if (s.isEmpty()) {
            s.push(x);
            minE = x;
        } else if (x < minE) {
            s.push(x - minE);
            minE = x;
        } else {
            s.push(x);
        }

    }

    public int pop() {

        if (s.isEmpty()) {
            System.out.println("empty");
            return Integer.MIN_VALUE;
        }

        int x = s.pop();

        if (x < minE) {
            minE = minE - x;
            x = (x + minE);
        }

        return x;

    }

    public int peek() {

        if (s.isEmpty()) {
            System.out.println("empty");
            return Integer.MIN_VALUE;
        }

        int x = s.peek();

        if (x < minE) {
            int prevMin = minE - x;
            x = prevMin + x;
        }

        return x;

    }

    public int getMin() {
        return minE;
    }

}
