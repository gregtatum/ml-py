import type { State as MainState } from 'src/store/reducers';
import type { PlainInternal } from 'src/store/thunks';
import * as Plain from 'src/store/plain';

export type State = MainState;

/**
 * Re-exports.
 */
// export { AppState, State } from 'src/store/reducers';

/**
 * Selectors always take the root state, and return some part of it.
 */
export type Selector<Returns> = (state: State) => Returns;

/**
 * A utility function to extract the values from an Object.
 */
export type Values<T> = T[keyof T];

/**
 * Action creators return an action object. This utility type extracts the returned
 * type so that the actions don't need to be manually typed.
 */
type ExtractActions<T extends { [key: string]: (...args: any) => any }> =
  Values<{
    [FnName in keyof T]: ReturnType<T[FnName]>;
  }>;

/**
 * Automatically extract the action object from the action creators.
 */
export type Action =
  | ExtractActions<typeof Plain>
  | ExtractActions<PlainInternal>;

/**
 * Provide a mechanism to easily define reducers that are bound to the current
 * set of Actions, and enforce the constraint that the first parameter must be
 * the same as the return value.
 *
 * See src/reducers for practical examples of how this is used.
 */
export type Reducer<S> = (state: S, action: Action) => S;

export type Thunk<Returns = void> = (
  dispatch: Dispatch,
  getState: () => State,
) => Returns;

// type DeThunkFn<T extends Thunk<Returns>, Returns> = (thunk: T) => Returns;

// export type DeThunk<
//   T extends Thunk<Returns>,
//   Returns = ReturnType<ReturnType<Thunk>>
// > = ReturnType<
//   DeThunkFn<T, ReturnType<ReturnType<Returns>>
// >;

/**
 * The rest of these pre-fill Redux with all of the configured Actions and middlewares.
 */
type ThunkDispatch = <Returns>(action: Thunk<Returns>) => Returns;
type PlainDispatch = (action: Action) => Action;
export type GetState = () => State;
export type Dispatch = PlainDispatch & ThunkDispatch;
export type Store = {
  dispatch: Dispatch;
  getState(): State;
  subscribe(listener: () => void): unknown;
  replaceReducer(nextReducer: Reducer<State>): void;
};
