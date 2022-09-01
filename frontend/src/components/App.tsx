import * as React from 'react';

import './App.css';
import { Mnist } from './Mnist';

export function App() {
  return (
    <div>
      <h1>Draw a number</h1>
      <Mnist />
    </div>
  );
}
