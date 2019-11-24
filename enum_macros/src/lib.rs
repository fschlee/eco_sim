extern crate proc_macro;

use proc_macro::{TokenStream, Ident};
use quote::quote;


#[proc_macro_derive(EnumCount)]
pub fn enum_count(input: TokenStream) -> TokenStream {
    let ast : syn::DeriveInput = syn::parse(input).unwrap();
    let mut count = 0usize;
    let en  = match ast.data {
        syn::Data::Enum(en) => en,
        _ => panic!("Count only works for enum types"),
    };
    let mut match_arms = Vec::new();
    for variant in en.variants {
        let ident = variant.ident;
        match_arms.push(quote!{#ident => #count, });
        count += 1;
    }
    let name = ast.ident;
    let gen = quote! {
        impl Count for #name {
           const COUNT : usize = #count;
           fn idx(&self)-> usize {
               use #name::*;
               match self {
                   #(#match_arms)*
               }
           }
        }
    };
    gen.into()
}

#[proc_macro_derive(EnumIter)]
pub fn enum_iter(input: TokenStream) -> TokenStream {
    let ast : syn::DeriveInput = syn::parse(input).unwrap();
    let mut count = 0usize;
    let en  = match ast.data {
        syn::Data::Enum(en) => en,
        _ => panic!("Count only works for enum types"),
    };
    let mut match_arms = Vec::new();
    for variant in en.variants {
        let ident = variant.ident;
        match_arms.push(quote!{ #count => Some(#ident), });
        count += 1;
    }
    let name = ast.ident;
    let struct_name : syn::Ident = syn::parse_str(&format!("{}Iterator", name)).unwrap();
    let gen = quote! {
        pub struct #struct_name(usize);

        impl Iterator for #struct_name {
            type Item = #name;
            fn next(&mut self) -> Option<Self::Item> {
                use #name::*;
                let res = match self.0 {
                    #(#match_arms)*
                    _ => None,
                };
                if res.is_some() {
                    self.0 += 1;
                }
                res
            }
        }
        impl #name {
            pub fn iter() -> #struct_name {
                #struct_name(0)
            }
        }
    };
    gen.into()
}