package com.rsclabs;

import java.util.Random;


/*
    Example of Linear regression solver in java.

    The model will attempt to solve the function

    f(x) = a * x + b

    But what does that mean? This is the the function for a point on a line.
    Plugging the numbers a=8 and b=3 any value of x will create a point on a line.

    Imagine we didn't know what a and b were, but we had a data set for lots of values of x.

    This is a linear regression model that will use stochastic gradient descent
    to 'fit' the variables a and b to the dataset.
*/
public class LinearModel
{
    public static double linearFunction(double a, double b, double x)
    {
        return a * x + b;
    }

    // generates a set of random numbers, used for creating test data
    public static double[] randNums(int number)
    {
        Random r = new Random();
        double nums[] = new double[number];

        for(int i =0; i < number; ++i)
        {
            nums[i] = r.nextDouble();
        }

        return nums;
    }

    // generate Y's
    public static double[] makeY(double a, double b, double x[])
    {
        double y[] = new double[x.length];

        for(int i = 0; i < x.length; ++i)
        {
            y[i] = linearFunction(a,b,x[i]);
        }

        return y;
    }

    // this is our loss function, it measures how good our model is
    public static double sumOfSquaredErrors(double y[], double pred[])
    {
        double total = 0;

        for(int i = 0; i < y.length; ++i)
        {
            total += (y[i] - pred[i]) * (y[i] - pred[i]);
        }

        return total;
    }

    public static double loss(double y[], double a, double b, double x[])
    {
        return sumOfSquaredErrors(y,makeY(a,b,x));
    }

    public static double averageLoss(double y[], double a, double b, double x[])
    {
        return Math.sqrt(loss(y,a,b,x) / (double)x.length);
    }

    public static double mean(double[] m)
    {
        double sum = 0;

        for (int i = 0; i < m.length; ++i)
        {
            sum += m[i];
        }

        return sum / m.length;
    }

    public static void main(String args[])
    {
        // set these numbers to anything and the model will try to guess them
        double a=3;
        double b=8;

        // next we generate 'some' data using our known values of a and b
        double x[] = randNums(30);
        double y[] = makeY(a,b,x);

        // we set the guess of a and b to some random number, it doesn't matter what
        double a_guess = -5;
        double b_guess = -3;

        double loss = averageLoss(y,a_guess, b_guess,x);

        System.out.println("Starting loss :"+loss);

        // learning rate is how quickly gradient descent will move towards it's final target
        double learningRate = 0.01;

        // in order for gradient descent to work we need to have the derivative of our loss function (with respect to our unknowns).
        // If we look above we can see our loss (sumOfSquaredErrors above) function is
        //
        // lossFunc = (y - predictions)^2
        //
        // to expand
        //
        // lossFunc = (y - (a * x + b )) ^ 2
        //
        // we want to calcualte dy/da and dy/db
        //
        // if we were doing this in python there are ways of automatically calculating the derivatives.
        // We could spend time working this out, however wolfram alpha will do this for us
        //
        // here's what to type into wolfram
        //
        // d[(y-(a*x+b))^2,b] = 2 (b + a x - y)      = 2 (y_pred - y)
        // d[(y-(a*x+b))^2,a] = 2 x (b + a x - y)    = x * dy/db
        //
        // final part x * dy/db is just a short cut, you could expand it to full for you wanted, but you'd just be calculating more than you need to.
        // Oh on wolfram you'll notice the derivatives are negative, you might wonder where that's gone. Notice we're subtracting dy/da from our guess.
        //
        // below is the gradient descent update loop

        double dydb[] = new double[x.length];
        double dyda[] = new double[x.length];

        // the higher this number the more accurate the guess will be
        for( int i = 0; i < 3000; ++i )
        {
            double y_pred[] = makeY(a_guess, b_guess, x);

            for( int j = 0; j < x.length; ++j )
            {
                dydb[j] = 2 * (y_pred[j] - y[j]);
                dyda[j] = x[j] * dydb[j];
            }

            a_guess -= learningRate*mean(dyda);
            b_guess -= learningRate*mean(dydb);

            if( i % 100 == 0 )
            {
                System.out.println("loss = " + averageLoss(y, a_guess, b_guess, x));
            }
        }

        System.out.println("I guess a="+a_guess+" and b="+b_guess);
    }
}
