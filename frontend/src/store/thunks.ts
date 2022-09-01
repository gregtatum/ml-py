import { Thunk } from 'src/@types';
import * as Plain from './plain';

/**
 * This file contains all of the thunk actions, that contain extra logic,
 * such as conditional dispatches, and multiple async calls.
 */

/**
 * These should only be used internally in thunks.
 */
const PlainInternal = {
  example() {
    return { type: 'example' as const };
  },
};

export type PlainInternal = typeof PlainInternal;

export function exampleThunk(): Thunk {
  return async (dispatch, getState) => {
    console.log(getState());
    dispatch(PlainInternal.example());
    dispatch(Plain.example());
  };
}
